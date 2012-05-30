
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
        STR_CASE(CV_8S);
        STR_CASE(CV_16U);
        STR_CASE(CV_16S);
        STR_CASE(CV_32S);
        STR_CASE(CV_32F);
        STR_CASE(CV_64F);
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

Mat load_sift_descriptors(const string& image_name,
                          std::vector<KeyPoint>& keypoints,
                          Mat& image) {
    SIFT sift;
    Mat descriptors;
    image = imread(image_name, 1);
    if (!image.data) {
        throw std::runtime_error(image_name + " failed to load");
    }
    sift(image, noArray(), keypoints, descriptors, false);
    return descriptors;
}

Mat load_sift_descriptors(const string& image_name) {
    std::vector<KeyPoint> keypoints;
    Mat image;

    return load_sift_descriptors(image_name, keypoints, image);
}

Mat load_sift_descriptors(const FileSet& images) {
    Mat accum;
    for (const std::string& image_name : images) {
        Mat descriptors = load_sift_descriptors(image_name);
        cout << OUT(descriptors.rows) << endl;

        if (accum.empty()) {
            accum = descriptors;
        } else {
            Mat tmp;
            vconcat(accum, descriptors, tmp);
            accum = tmp;
        }
    }

    return accum;
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

void draw_clustered_keypoints(const ExtEM& em,
                              const string& file_name) {
    std::vector<KeyPoint> keypoints;
    Mat image;
    Mat img_des = load_sift_descriptors(file_name, keypoints, image);
    std::vector<Scalar> colors = {{0,0,0,255}, {255,0,0,255}, {0,255,0,255},
                                  {0,0,255,255}, {255,255,255,255},
                                  {150,150,0,255}, {0,150,150,255},
                                  {150, 0,150,255}};

    for (int ii = 0; ii < img_des.rows; ++ii) {
        Mat ex_out = em.predict_ex(img_des.row(ii));
        drawKeypoints(image, fn::slice(keypoints, ii, ii + 1), image,
                      colors[max_idx(ex_out)],
                      DrawMatchesFlags::DRAW_OVER_OUTIMG |
                      DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    }
    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
    imshow( "Display Image", image );
    waitKey(0);
}

void extract_nn_matrix(const ExtEM& em,
                       const FileSet& images,
                       size_t& res_idx,
                       Mat& result) {
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx, ++res_idx) {
        Mat img_des = load_sift_descriptors(images[img_idx]);
        aggregate_predictions(em, img_des).copyTo(result.row(res_idx));
    }
}

void learn_training_matrix(const ExtEM& em,
                           const FileSet& train_pos,
                           const FileSet& train_neg,
                           Mat& train_input,
                           Mat& train_outputs) {

    //draw_clustered_keypoints(em, "bikes/0001.jpg");

    train_input = Mat::zeros(train_pos.size() + train_neg.size(),
                             em.getInt("nclusters"), CV_32FC1);
    train_outputs = Mat::zeros(train_pos.size() + train_neg.size(),
                               1, CV_32FC1);

    size_t res_idx = 0;

    train_outputs.rowRange(0, train_pos.size()) = 1;
    extract_nn_matrix(em, train_pos, res_idx, train_input);
    train_outputs.rowRange(train_pos.size(), train_outputs.rows) = Scalar(0);
    extract_nn_matrix(em, train_neg, res_idx, train_input);

    /*
    for (size_t img_idx = 0; img_idx < train_pos.size(); ++img_idx, ++res_idx) {
        Mat img_des = load_sift_descriptors(train_pos[img_idx]);
        aggregate_predictions(em, img_des).copyTo(train_input.row(res_idx));
        train_outputs.at<float>(res_idx) = 1;
        }*/
/*
    for (size_t img_idx = 0; img_idx < train_neg.size(); ++img_idx, ++res_idx) {
        Mat img_des = load_sift_descriptors(train_neg[img_idx]);
        aggregate_predictions(em, img_des).copyTo(train_input.row(res_idx));
        train_outputs.at<float>(res_idx) = 0;
    }*/
}

void generate_testing_matrix(const ExtEM& em,
                             const FileSet& test_pos,
                             const FileSet& test_neg,
                             Mat& test_input,
                             Mat& test_output) {
    test_input = Mat::zeros(test_pos.size() + test_neg.size(),
                            em.getInt("nclusters"), CV_32FC1);
    test_output = Mat::zeros(test_pos.size() + test_neg.size(),
                              1, CV_32FC1);

    size_t res_idx = 0;
    test_output.rowRange(0, test_pos.size()) = 1;
    extract_nn_matrix(em, test_pos, res_idx, test_input);
    extract_nn_matrix(em, test_neg, res_idx, test_input);
}
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

    int num_train_correct = countNonZero((output > 0.5) == (predicted_output > 0.5));
    cout << OUT(num_train_correct) << endl;
    cout << OUT(output.rows) << endl;

    cout << OUT(float(num_train_correct) / float(output.rows)) << endl;
}

// objrec positive_dir negative_dir...
int main( int argc, char** argv )
{
    if (argc != 3) {
        usage();
        return 1;
    }

    FileSet pos_examples = glob_ex(std::string(argv[1]) + "/*.jpg");
    FileSet pos_train_examples;
    FileSet pos_test_examples;
    split_set(pos_examples, 0.66, pos_train_examples, pos_test_examples);

    FileSet neg_examples = glob_ex(std::string(argv[2]) + "/*.jpg");
    FileSet neg_train_examples;
    FileSet neg_test_examples;
    split_set(neg_examples, 0.66, neg_train_examples, neg_test_examples);

    // TESTING: maximum training size for testing peformance.
    // remove when done testing...
    pos_train_examples = fn::slice(pos_train_examples, 0, 200);
    neg_train_examples = fn::slice(neg_train_examples, 0, 200);

    pos_test_examples = fn::slice(pos_test_examples, 0, 200);
    neg_test_examples = fn::slice(neg_test_examples, 0, 200);

    /*
    cout << OUT(pos_examples.size()) << endl;
    cout << OUT(pos_train_examples.size()) << endl;
    cout << OUT(pos_train_examples) << endl;
    cout << OUT(pos_test_examples.size()) << endl;
    cout << OUT(pos_test_examples) << endl;*/

    Mat descriptors = load_sift_descriptors(pos_train_examples);
    /*
    cout << OUT(descriptors.rows) << endl;
    cout << OUT(descriptors.rowRange(0, 5)) << endl;
    cout << OUT(descriptors.rowRange(descriptors.rows - 5, descriptors.rows)) << endl;*/


    // opencv neural networks want 32 bit floats.
    Mat converted_descs;
    descriptors.convertTo(converted_descs, CV_32FC1);

    cout << OUT(converted_descs.size()) << endl;
    //cout << OUT(converted_descs.rowRange(0,5)) << endl;

    // Learn EM.
    int sift_clusters = 10;
    cout << "START: em training" << endl;
    ExtEM em(sift_clusters, EM::COV_MAT_GENERIC);

    if (!em.train(converted_descs)) {
        cout << "error training" << endl;
        exit(1);
    }

    cout << "STOP: em training" << endl;

    cout << OUT(em.isTrained()) << endl;
    cout << OUT(em.getMat("means")) << endl;
    //em.diag();


    cout << "constructing training matrix" << endl;
    Mat nn_train_input;
    Mat nn_train_output;
    learn_training_matrix(em,
                          pos_train_examples,
                          neg_train_examples,
                          nn_train_input,
                          nn_train_output);

    cout << "constructing training matrix finished" << endl;

    cout << OUT(nn_train_input) << endl;
    cout << OUT(nn_train_output) << endl;

    cout << OUT(nn_train_input.cols) << endl;
    Mat layers = Mat(Matx<int, 1, 3>(nn_train_input.cols, nn_train_input.cols, 1));
    cout << OUT(layers) << endl;
    cout << OUT(type_str(layers.type())) << endl;
    CvANN_MLP net(layers);

    cout << "training neural net";
    int num_iters = net.train(nn_train_input, nn_train_output, Mat()/*,
                              Mat(), CvANN_MLP_TrainParams(),
                              CvANN_MLP::NO_INPUT_SCALE*/);

    cout << "training neural net finished";
    cout << OUT(num_iters) << endl;
    
    cout << "training set accuracy:" << endl;
    test_nn(net, nn_train_input, nn_train_output);

    // Set up testing set matrices
    Mat nn_test_input;
    Mat nn_test_outputs;

    generate_testing_matrix(em,
                            pos_test_examples,
                            neg_test_examples,
                            nn_test_input,
                            nn_test_outputs);

    cout << "testing set accuracy:" << endl;
    test_nn(net, nn_test_input, nn_test_outputs);
}
