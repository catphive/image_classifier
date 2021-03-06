
#include "loader.hpp"
#include "util.hpp"
#include "files.hpp"

#include <stddef.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include <stdexcept>
#include <string>

#include <getopt.h>

using namespace cv;

using std::cout;
using std::cerr;
using std::endl;
using std::string;

std::ostream& operator<<(std::ostream& out, const cv::Size& size) {
    return out << "(" << size.height << ", " << size.width << ")";
}

std::ostream& operator<<(std::ostream& out, const cv::KeyPoint& keypoint) {
    return out << "(" << keypoint.pt << ", " << keypoint.size << ", "
               << keypoint.angle << ", " << keypoint.response << ", "
               << keypoint.octave << ", " << keypoint.class_id << ")";
}

std::ostream& operator<<(std::ostream& out, const cv::Scalar& scalar) {
    return out << "(" << scalar[0] << ", " << scalar[1] << ", "
               << scalar[2] << ", " << scalar[3] << ")";
}

/*
  Returns text name of opencv array type integer.
  type: an opencv matrix element type.
 */
std::string type_str(int type) {

    switch(type) {
        STR_CASE(CV_8U);
        STR_CASE(CV_8UC2);
        STR_CASE(CV_8UC3);
        STR_CASE(CV_8UC4);
        STR_CASE(CV_8S);
        STR_CASE(CV_8SC2);
        STR_CASE(CV_8SC3);
        STR_CASE(CV_8SC4);
        STR_CASE(CV_16U);
        STR_CASE(CV_16S);
        STR_CASE(CV_32S);
        STR_CASE(CV_32SC2);
        STR_CASE(CV_32SC3);
        STR_CASE(CV_32SC4);
        STR_CASE(CV_32F);
        STR_CASE(CV_32FC2);
        STR_CASE(CV_32FC3);
        STR_CASE(CV_32FC4);
        STR_CASE(CV_64F);
        STR_CASE(CV_64FC2);
        STR_CASE(CV_64FC3);
        STR_CASE(CV_64FC4);
    default:
        return fn::format("unknown type %d", type);
    }
}

// This derived class works around some limitations in opencv's EM
// implementation.
// Specifically, opencv's EM implementation returns posterior probabilities,
// whereas this classifier is only interested in joint probabilities.
struct ExtEM : cv::EM {
    ExtEM(int nclusters=EM::DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL,
          const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+
                                                    TermCriteria::EPS,
                                                    EM::DEFAULT_MAX_ITERS, FLT_EPSILON)):
        EM(nclusters, covMatType, termCrit) {
    }
    
    void diag() {
        
        cout << "Covariant matrices:" << endl;
        for (auto& cov : covs) {
            cout << cov << endl;
        }
    }

    Mat predict_ex(InputArray _sample) const {
        Mat sample = _sample.getMat();

        CV_Assert(isTrained());

        CV_Assert(!sample.empty());
        if(sample.type() != CV_64FC1) {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }

        Mat joint_probs;

        compute_joint_prob(sample, joint_probs);

        return joint_probs;
    }
    
    // This is a modified version of opencv's stock computeProbabilities function.
    // Opencv's EM implementation only retrieves posterior probability,
    // which would not be appropriate the generative/descriminative method.
    // This this modified function extracts the joint probability instead.
    // Aside from that, no changes were made to the function.
    void compute_joint_prob(const Mat& sample, Mat& joint_probs) const
    {
        // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
        // q = arg(max_k(L_ik))
        // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
        // see Alex Smola's blog http://blog.smola.org/page/2 for
        // details on the log-sum-exp trick

        CV_Assert(!means.empty());
        CV_Assert(sample.type() == CV_64FC1);
        CV_Assert(sample.rows == 1);
        CV_Assert(sample.cols == means.cols);

        int dim = sample.cols;

        Mat L(1, nclusters, CV_64FC1);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++) {
            const Mat centeredSample = sample - means.row(clusterIndex);
            
            Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
                centeredSample : centeredSample * covsRotateMats[clusterIndex];
            
            double Lval = 0;
            for(int di = 0; di < dim; di++) {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;
        }

        joint_probs.create(1, nclusters, CV_64FC1);
        L.copyTo(joint_probs);
    }
};

typedef Ptr<ExtEM> ExtEMPtr;

// Training data for neural networks.
struct NNData {
    Mat nn_input;
    Mat nn_output;
};

typedef vector<NNData> NNDataVec;

// Model for a feature, including statistical model and function for
// extracting features from images.
struct FeatureModel {
    LoaderPtr loader;
    ExtEMPtr em;
};

// Returns the index of the maximum element of an opencv vector.
size_t max_idx(const Mat& vec) {
    double max = -DBL_MAX;
    size_t max_idx = 0;
    for (size_t idx = 0; idx < vec.total(); ++idx) {
        double cur = vec.at<double>(idx);
        if (cur > max) {
            max = cur;
            max_idx = idx;
        }
    }
    return max_idx;
}

void usage() {
    cerr << "USAGE: objrec [-f n] positive_example_dir negative_example_dir ...\\" 
         << "    -f n  specifies index of feature to use. If not specified\n"
         << "          use all feature types at once.\n"
         << "          -f 0 selects only SIFT.\n"
         << "          -f 1 selects only CIA L*a*b* color."
         << endl;
}

/* Determine how well descriptors match against the components of em.
   Return an opencv row vector of the maximum matchs for each
   component.  Note that by "match" I'm referring to the joint
   probability of a feature and a component gaussian.
*/
Mat aggregate_predictions(const ExtEM& em,
                          const Mat& descriptors) {
    Mat predictions(descriptors.rows, em.getInt("nclusters"), CV_64F);
    for (int row_idx = 0; row_idx < descriptors.rows; ++row_idx) {
        Mat tmp = em.predict_ex(descriptors.row(row_idx));
         tmp.copyTo(predictions.row(row_idx));
     }

    Mat aggr;
    reduce(predictions, aggr, 0, CV_REDUCE_MAX);

    Mat result;
    aggr.convertTo(result, CV_32FC1);

    return result;
}

/*
  Extract a set of aggregated joint probabilities of image features
  with respect to a guassian mixture model.
  em: A gaussian mixture model.
  images: A set of images.
  loader: An object that extracts features from images.
  res_idx: row of result to write to. Increments res_idx after writing each row.
  result: output matrix.
 */
void extract_nn_matrix(const ExtEM& em,
                       const FileSet& images,
                       LoaderPtr loader,
                       size_t& res_idx,
                       Mat& result) {
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx, ++res_idx) {
        Mat img_des = loader->load(images[img_idx]);
        aggregate_predictions(em, img_des).copyTo(result.row(res_idx));
    }
}

/*
  Extract neural network training data from a mixture model and
  positive and negative training examples.
  em: A gaussian mixture model.
  train_pos: positive training examples.
  train_neg: negative training examples.
  loader: Responsible for extracting the correct type of feature.
 */
NNData extract_nn_data(const ExtEM& em,
                       const FileSet& train_pos,
                       const FileSet& train_neg,
                       LoaderPtr loader) {
    NNData data;

    data.nn_input = Mat::zeros(train_pos.size() + train_neg.size(),
                             em.getInt("nclusters"), CV_32FC1);
    data.nn_output = Mat::zeros(train_pos.size() + train_neg.size(),
                               1, CV_32FC1);

    size_t res_idx = 0;

    data.nn_output.rowRange(0, train_pos.size()) = 1;
    extract_nn_matrix(em, train_pos, loader, res_idx, data.nn_input);
    data.nn_output.rowRange(train_pos.size(), data.nn_output.rows) = Scalar(0);
    extract_nn_matrix(em, train_neg, loader, res_idx, data.nn_input);
    return data;
}

/*
  Return the number of false positives by comparing output and predicted_output.
 */
int false_positives(const Mat& output,
                    const Mat& predicted_output) {
    assert(output.rows == predicted_output.rows);

    int accum = 0;
    for (int idx = 0; idx < output.rows; ++idx) {
        if (predicted_output.at<uchar>(idx) &&
            !output.at<uchar>(idx)) {
            ++accum;
        }
    }

    return accum;
}

/*
  Returns all false positives from imeages by comparing output and
  predicted_output.
 */
FileSet collect_false_positives(const Mat& output,
                                const Mat& predicted_output,
                                const FileSet& images) {
    assert(output.rows == predicted_output.rows);
    FileSet result;

    for (int idx = 0; idx < output.rows; ++idx) {
        if (predicted_output.at<uchar>(idx) &&
            !output.at<uchar>(idx)) {
            result.push_back(images[idx]);
        }
    }

    return result;
}

/*
  Returns all false negatives from imeages by comparing output and
  predicted_output.
 */
FileSet collect_false_negatives(const Mat& output,
                                const Mat& predicted_output,
                                const FileSet& images) {
    assert(output.rows == predicted_output.rows);
    FileSet result;

    for (int idx = 0; idx < output.rows; ++idx) {
        if (!predicted_output.at<uchar>(idx) &&
            output.at<uchar>(idx)) {
            result.push_back(images[idx]);
        }
    }

    return result;
}

/*
  Return the number of false negatives by comparing output and predicted_output.
 */
int false_negatives(const Mat& output,
                    const Mat& predicted_output) {
    assert(output.rows == predicted_output.rows);

    int accum = 0;
    for (int idx = 0; idx < output.rows; ++idx) {
        if (!predicted_output.at<uchar>(idx) &&
            output.at<uchar>(idx)) {
            ++accum;
        }
    }

    return accum;
}

/*
  Returns all correctly classified images by comparing output to
  predicted output.
 */
FileSet collect_correct(const Mat& output,
                        const Mat& predicted_output,
                        const FileSet& images) {
    assert(output.rows == predicted_output.rows);
    FileSet result;

    for (int idx = 0; idx < output.rows; ++idx) {
        if (predicted_output.at<uchar>(idx) == output.at<uchar>(idx)) {
            result.push_back(images[idx]);
        }
    }

    return result;
}

/*
  Collect and print accuracy of neural network net the provided input
  and output. If images is not NULL, also printed false positives,
  false negatives, and some example correct classifications.
 */
void test_nn(const CvANN_MLP& net, const Mat& input, Mat output, FileSet* images) {
    Mat predicted_output;
    net.predict(input, predicted_output);
    cout << OUT(predicted_output) << endl;
    output = output > 0.5;
    cout << OUT(output) << endl;
    predicted_output = predicted_output > 0.5;
    cout << OUT(predicted_output) << endl;

    cout << OUT(output == predicted_output) << endl;

    int num_correct = countNonZero(output == predicted_output);
    cout << OUT(num_correct) << endl;
    cout << OUT(output.rows) << endl;

    cout << OUT(float(num_correct) / float(output.rows)) << endl;
    
    int false_pos = false_positives(output, predicted_output);
    cout << "false positives: " << false_pos
         << "(" << float(false_pos) / float(output.rows) << ")" << endl;

    if (images) {
        cout << collect_false_positives(output, predicted_output, *images)
             << endl;
    }

    int false_neg = false_negatives(output, predicted_output);
    cout << "false negatives: " << false_neg
         << "(" << float(false_neg) / float(output.rows) << ")" << endl;

    if (images) {
        cout << collect_false_negatives(output, predicted_output, *images)
             << endl;
    }

    if (images) {
        cout << "sample correct images: "
             << fn::slice(collect_correct(output, predicted_output, *images),
                          0, 5)
             << endl;
    }
}

/*
  Train and return a gaussian mixture model with the specified number
  of clusters from the provided descriptors.
 */
Ptr<ExtEM> train_em(int clusters, const Mat& descriptors) {
    Ptr<ExtEM> em = new ExtEM(clusters/*, EM::COV_MAT_GENERIC*/);

    if (!em->train(descriptors)) {
        cout << "error training" << endl;
        exit(1);
    }

    cout << OUT(em->isTrained()) << endl;
    cout << OUT(em->getMat("means")) << endl;

    return em;
}

/*
  Learn a model from the positive training example images using loader
  to extract features.
 */
FeatureModel learn_model(const FileSet& pos_train_examples,
                         LoaderPtr loader) {
    FeatureModel model;
    model.loader = loader;

    cout << "loading descriptors" << endl;
    Mat descriptors = loader->load_set(pos_train_examples);
    cout << "finished loading descriptors" << endl;

    cout << "START: converting descriptors" << endl;
    Mat converted_descs;
    descriptors.convertTo(converted_descs, CV_32FC1);
    cout << "STOP: converting descriptors" << endl;

    cout << OUT(converted_descs.size()) << endl;

    // Learn EM.
    cout << "START: em training" << endl;
    model.em = train_em(loader->em_clusters(), converted_descs);
    cout << "STOP: em training" << endl;

    return model;
}

/*
  Append mat_in to the right side of mat_out
*/
void append_horiz(const Mat& mat_in, Mat& mat_out) {
    if (mat_out.empty()) {
        mat_out = mat_in;
    } else {
        Mat tmp;
        hconcat(mat_out, mat_in, tmp);
        mat_out = tmp;
    }
}

/*
  Combine the neural network training data in data_vec and return it.
 */
NNData combine_data(const NNDataVec& data_vec) {
    NNData result;

    for (const NNData& data : data_vec) {
        if (result.nn_output.empty()) {
            result.nn_output = data.nn_output;
        }
        append_horiz(data.nn_input, result.nn_input);
    }

    return result;
}

// objrec [-f n] positive_dir negative_dir...
int main( int argc, char** argv )
{
    static struct option long_options[] = {
        {"help",           required_argument, NULL, 'h'},
        {"only-feature",   required_argument, NULL, 'f'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int ch = 0;
    int only_feature = -1;
    while ((ch = getopt_long (argc, argv, "f:",
                              long_options, &option_index)) != -1) {
        switch (ch) {
        case 'h':
            usage();
            return 0;
        case 'f':
            only_feature = atoi(optarg);
            break;
        case '?':
            if (isprint (optopt)) {
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            }
            usage();
            return 1;
        default:
            abort ();
        }
    }

    if (argc - optind < 3) {
        usage();
        return 1;
    }

    FileSet pos_examples = glob_ex(std::string(argv[optind]) + "/*.jpg");
    FileSet pos_train_examples;
    FileSet pos_test_examples;
    split_set(pos_examples, 0.66, pos_train_examples, pos_test_examples);

    FileSet neg_train_examples;
    FileSet neg_test_examples;

    for (int arg_idx = optind + 1; arg_idx < argc; ++arg_idx) {
        FileSet neg_examples = glob_ex(std::string(argv[arg_idx]) + "/*.jpg");
        split_set(neg_examples, 0.66, neg_train_examples, neg_test_examples);
    }

    shuffle(neg_train_examples);
    shuffle(neg_test_examples);

    cout << OUT(neg_train_examples.size()) << endl;
    cout << OUT(neg_test_examples.size()) << endl;

    pos_train_examples = fn::slice(pos_train_examples, 0, 200);
    neg_train_examples = fn::slice(neg_train_examples, 0, 200);

    pos_test_examples = fn::slice(pos_test_examples, 0, 100);
    neg_test_examples = fn::slice(neg_test_examples, 0, 100);

    
    LoaderVec loaders = make_loaders();

    if (only_feature >= 0) {
        cout << "using only feature " << only_feature << endl;
        loaders = fn::slice(loaders, only_feature, only_feature + 1);
    }

    vector<FeatureModel> models;
    vector<NNData> train_data;
    for (LoaderPtr loader : loaders) {
        models.push_back(learn_model(pos_train_examples,
                                     loader));

        cout << "START: extract_nn_data" << endl;
        train_data.push_back(extract_nn_data(*models.back().em,
                                             pos_train_examples,
                                             neg_train_examples,
                                             loader));
        cout << "STOP: extract_nn_data" << endl;
    }

    NNData combined_train_data = combine_data(train_data);

    Mat layers = Mat(Matx<int, 1, 3>(combined_train_data.nn_input.cols,
                                     combined_train_data.nn_input.cols,
                                     1));
    cout << OUT(layers) << endl;
    cout << OUT(type_str(layers.type())) << endl;
    CvANN_MLP net(layers);

    cout << "START: training neural net";
    cout << OUT(type_str(combined_train_data.nn_output.type())) << endl;
    cout << OUT(combined_train_data.nn_output) << endl;
    int num_iters = net.train(combined_train_data.nn_input,
                              combined_train_data.nn_output,
                              Mat());

    cout << "STOP: training neural net finished";
    cout << OUT(num_iters) << endl;
    
    cout << "TRAINING SET ACCURACY:" << endl;
    test_nn(net, combined_train_data.nn_input, combined_train_data.nn_output, NULL);

    // Test testing data.
    NNDataVec test_data;
    for (const FeatureModel& model : models) {
        test_data.push_back((extract_nn_data(*model.em,
                                             pos_test_examples,
                                             neg_test_examples,
                                             model.loader)));
    }

    NNData combined_test_data = combine_data(test_data);

    cout << "TESTING SET ACCURACY:" << endl;

    FileSet all_test_examples = pos_test_examples;
    all_test_examples.insert(all_test_examples.end(),
                             neg_test_examples.begin(),
                             neg_test_examples.end());
    
    test_nn(net,
            combined_test_data.nn_input,
            combined_test_data.nn_output,
            &all_test_examples);
}
