
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

/*
  1. load images in (matching) directory
  2. extract features
  3. train EM.
 */

// assumes column vectors for mean and sample.
double mv_log_gaussian(const Mat& cov, const Mat& mean, const Mat& sample) {
    CV_Assert(mean.cols == 1);
    CV_Assert(sample.cols == 1);

    double det = determinant(cov);
    cout << OUT(det) << endl;
    Mat inv;
    invert(cov, inv, DECOMP_SVD);

    cout << OUT(cov.cols) << endl;
    
    double denom = std::pow(double(2 * M_PI), double(cov.cols) / 2) * sqrt(det);
    cout << OUT(denom) << endl;

    Mat centered = sample - mean;
    double exp_term = Mat(-0.5 * centered.t() * inv * centered).at<double>(0);

    cout << OUT(exp_term) << endl;

    return det;
}



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
        /*
        cout << "covsEigenValues" << endl;
        Matx<double, 1, 3> small_mat(1.1,2.2,3.3);
        cout << Mat(small_mat) << endl;
        cout << Mat::diag(Mat(small_mat)) << endl;
        
        
        for (size_t ii = 0; ii < covsEigenValues.size(); ++ii) {
            cout << OUT(type_str(covsEigenValues[ii].type())) << endl;
            cout << covsEigenValues[ii] << endl;
            //cout << Mat::diag(covsEigenValues[ii]) << endl;
            }*/
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
        for(int cluster_idx = 0; cluster_idx < nclusters; ++cluster_idx) {

            /*mv_gaussian(covs[cluster_idx],
                        means.row(cluster_idx).t(),
                        sample.t());*/
        }

        return joint_probs;
    }
    
    // This is a modified version of opencv's stock computeProbabilities function.
    // Opencv's EM implementation only retrieves posterior probability,
    // which would not be appropriate the generative/descriminative method.
    // This this modified function extracts the joint probability instead.
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

struct NNData {
    Mat nn_input;
    Mat nn_output;
};

typedef vector<NNData> NNDataVec;

struct FeatureModel {
    LoaderPtr loader;
    ExtEMPtr em;
};

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
    cerr << "USAGE: objrec positive_example_dir negative_example_dir" << endl;
}

Mat aggregate_predictions(const ExtEM& em,
                          const Mat& descriptors) {
    Mat aggr(0, descriptors.cols, CV_32FC1);
    for (int row_idx = 0; row_idx < descriptors.rows; ++row_idx) {
        aggr.push_back(em.predict_ex(descriptors.row(row_idx)));
    }

    Mat result;
    reduce(aggr, result, 0, CV_REDUCE_MAX);
    return result;
}



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

NNData extract_nn_data(const ExtEM& em,
                     const FileSet& train_pos,
                     const FileSet& train_neg,
                     LoaderPtr loader) {
    NNData data;

    //draw_clustered_keypoints(em, "bikes/0001.jpg");

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
void generate_testing_matrix(const ExtEM& em,
                             const FileSet& test_pos,
                             const FileSet& test_neg,
                             LoaderPtr loader,
                             Mat& test_input,
                             Mat& test_output) {
    test_input = Mat::zeros(test_pos.size() + test_neg.size(),
                            em.getInt("nclusters"), CV_32FC1);
    test_output = Mat::zeros(test_pos.size() + test_neg.size(),
                              1, CV_32FC1);

    size_t res_idx = 0;
    test_output.rowRange(0, test_pos.size()) = 1;
    extract_nn_matrix(em, test_pos, loader, res_idx, test_input);
    extract_nn_matrix(em, test_neg, loader, res_idx, test_input);
    }*/
/*
  1. load positive training/testing sets
  2. cluster based on positive training
  3. load negative training/testing sets
  4. build training matrix based on positive/negativ training
  5. train neural net
  6. using posive/negative testing sets for testing.
 */

void test_nn(const CvANN_MLP& net, const Mat& input, const Mat& output) {
    Mat predicted_output;
    net.predict(input, predicted_output);
    cout << OUT(output > 0.5) << endl;
    cout << OUT(predicted_output > 0.5) << endl;

    cout << OUT((output > 0.5) == (predicted_output > 0.5)) << endl;

    int num_correct = countNonZero((output > 0.5) == (predicted_output > 0.5));
    cout << OUT(num_correct) << endl;
    cout << OUT(output.rows) << endl;

    cout << OUT(float(num_correct) / float(output.rows)) << endl;
}

Ptr<ExtEM> train_em(int clusters, const Mat& descriptors) {
    Ptr<ExtEM> em = new ExtEM(clusters, EM::COV_MAT_GENERIC);

    if (!em->train(descriptors)) {
        cout << "error training" << endl;
        exit(1);
    }

    cout << OUT(em->isTrained()) << endl;
    cout << OUT(em->getMat("means")) << endl;

    return em;
}

//TODO: most of these parameters aren't necessary anymore.
FeatureModel learn_model(int em_clusters,
                         const FileSet& pos_train_examples,
                         const FileSet& pos_test_examples,
                         const FileSet& neg_train_examples,
                         const FileSet& neg_test_examples,
                         LoaderPtr loader) {
    FeatureModel model;
    model.loader = loader;

    cout << "loading descriptors" << endl;
    Mat descriptors = loader->load_set(pos_train_examples);
    cout << "finished loading descriptors" << endl;

    // TODO: is this really necessary? seems so for SIFT...
    Mat converted_descs;
    descriptors.convertTo(converted_descs, CV_32FC1);

    cout << OUT(converted_descs.size()) << endl;

    // Learn EM.
    cout << "START: em training" << endl;
    model.em = train_em(em_clusters, converted_descs);
    cout << "STOP: em training" << endl;

    return model;
}

void append_horiz(const Mat& mat_in, Mat& mat_out) {
    if (mat_out.empty()) {
        mat_out = mat_in;
    } else {
        Mat tmp;
        hconcat(mat_out, mat_in, tmp);
        mat_out = tmp;
    }
}

NNData combine_data(const NNDataVec& data_vec) {
    NNData result;

    // TODO: consider just creating the result matrix all at once.
    // This is not efficient.
    for (const NNData& data : data_vec) {
        if (result.nn_output.empty()) {
            result.nn_output = data.nn_output;
        }
        append_horiz(data.nn_input, result.nn_input);
    }

    return result;
}

// objrec positive_dir negative_dir...
int main( int argc, char** argv )
{
    if (argc < 3) {
        usage();
        return 1;
    }

    FileSet pos_examples = glob_ex(std::string(argv[1]) + "/*.jpg");
    FileSet pos_train_examples;
    FileSet pos_test_examples;
    split_set(pos_examples, 0.66, pos_train_examples, pos_test_examples);

    FileSet neg_train_examples;
    FileSet neg_test_examples;
    for (int arg_idx = 2; arg_idx < argc; ++arg_idx) {
        FileSet neg_examples = glob_ex(std::string(argv[arg_idx]) + "/*.jpg");
        split_set(neg_examples, 0.66, neg_train_examples, neg_test_examples);
    }

    shuffle(neg_train_examples);
    shuffle(neg_test_examples);

    cout << OUT(neg_train_examples.size()) << endl;
    cout << OUT(neg_test_examples.size()) << endl;

    // TESTING: maximum training size for testing peformance.
    // remove when done testing...
    pos_train_examples = fn::slice(pos_train_examples, 0, 200);
    neg_train_examples = fn::slice(neg_train_examples, 0, 200);

    pos_test_examples = fn::slice(pos_test_examples, 0, 100);
    neg_test_examples = fn::slice(neg_test_examples, 0, 100);

    
    LoaderVec loaders = make_loaders();
    vector<FeatureModel> models;
    vector<NNData> train_data;
    for (LoaderPtr loader : loaders) {
        models.push_back(learn_model(10,
                                     pos_train_examples,
                                     pos_test_examples,
                                     neg_train_examples,
                                     neg_test_examples,
                                     loader));

        cout << "constructing training matrix" << endl;
        train_data.push_back(extract_nn_data(*models.back().em,
                                             pos_train_examples,
                                             neg_train_examples,
                                             loader));
        cout << "constructing training matrix finishd" << endl;
    }

    NNData combined_train_data = combine_data(train_data);

    /*
    bool do_sift = true;
    bool do_color = true;

    Mat nn_train_input;
    Mat nn_train_output;

    if (do_sift) {
        train(10,
              pos_train_examples,
              pos_test_examples,
              neg_train_examples,
              neg_test_examples,
              load_sift_descriptors,
              load_sift_descriptor_set,
              nn_train_input
              nn_train_output);
    }

    if (do_color) {
        train(10,
              pos_train_examples,
              pos_test_examples,
              neg_train_examples,
              neg_test_examples,
              load_color_descriptors,
              load_color_descriptor_set,
              nn_train_input
              nn_train_output);
    }
    */
    /*
    cout << "loading SIFT descriptors" << endl;
    Mat sift_descriptors = load_sift_descriptor_set(pos_train_examples);
    cout << "finished loading SIFT descriptors" << endl;

    cout << "loading color descriptors" << endl;
    Mat color_descriptors = load_color_descriptor_set(pos_train_examples);
    cout << "finished loading color descriptors" << endl;

    // opencv neural networks want 32 bit floats.
    Mat converted_sift_descs;
    sift_descriptors.convertTo(converted_sift_descs, CV_32FC1);

    cout << OUT(converted_sift_descs.size()) << endl;
    cout << OUT(color_descriptors.size()) << endl;

    // Learn EM.

    cout << "START: em SIFT training" << endl;
    Ptr<ExtEM> sift_em = train_em(10, converted_sift_descs);
    cout << "STOP: em SIFT training" << endl;

    cout << "START: em color training" << endl;
    Ptr<ExtEM> color_em = train_em(10, color_descriptors);
    cout << "STOP: em color training" << endl;


    cout << "constructing training matrix" << endl;
    Mat nn_train_input;
    Mat nn_train_output;
    learn_training_matrix(*sift_em,
                          pos_train_examples,
                          neg_train_examples,
                          load_sift_descriptors,
                          nn_train_input,
                          nn_train_output);

    cout << "constructing training matrix finished" << endl;
    */

    /*
    cout << OUT(nn_train_input) << endl;
    cout << OUT(nn_train_output) << endl;
   

    cout << OUT(nn_train_input.cols) << endl;
    */

    // TODO: experiment with smaller hidden layer.
    Mat layers = Mat(Matx<int, 1, 3>(combined_train_data.nn_input.cols,
                                     combined_train_data.nn_input.cols,
                                     1));
    cout << OUT(layers) << endl;
    cout << OUT(type_str(layers.type())) << endl;
    CvANN_MLP net(layers);

    cout << "training neural net";
    cout << OUT(type_str(combined_train_data.nn_output.type())) << endl;
    cout << OUT(combined_train_data.nn_output) << endl;
    int num_iters = net.train(combined_train_data.nn_input,
                              combined_train_data.nn_output,
                              Mat());

    cout << "training neural net finished";
    cout << OUT(num_iters) << endl;
    
    cout << "training set accuracy:" << endl;
    test_nn(net, combined_train_data.nn_input, combined_train_data.nn_output);

    // Test testing data.
    NNDataVec test_data;
    for (const FeatureModel& model : models) {
        test_data.push_back((extract_nn_data(*model.em,
                                             pos_test_examples,
                                             neg_test_examples,
                                             model.loader)));
    }

    NNData combined_test_data = combine_data(test_data);

    /*
    // Set up testing set matrices
    Mat nn_test_input;
    Mat nn_test_outputs;

    generate_testing_matrix(*sift_em,
                            pos_test_examples,
                            neg_test_examples,
                            load_sift_descriptors,
                            nn_test_input,
                            nn_test_outputs);
    */

    cout << "testing set accuracy:" << endl;
    test_nn(net, combined_test_data.nn_input, combined_test_data.nn_output);
}
