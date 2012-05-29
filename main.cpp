
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

using namespace cv;

using std::cout;
using std::cerr;
using std::endl;

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

Mat load_sift_descriptors(const string& image_name) {
    SIFT sift;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat image = imread(image_name, 1);
    if (!image.data) {
        throw std::runtime_error(image_name + " failed to load");
    }
    sift(image, noArray(), keypoints, descriptors, false);
    return descriptors;
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
    
    //cout << "old: " << OUT(aggr) << endl;
    for (int row_idx = 0; row_idx < descriptors.rows; ++row_idx) {
        /*
        cout << OUT(descriptors.row(row_idx)) << endl;
        cout << OUT(descriptors.row(row_idx).cols) << endl;
        cout << OUT(type_str(descriptors.row(row_idx).type())) << endl;
        cout << OUT(aggr.row(row_idx)) << endl;
        cout << OUT(aggr.row(row_idx).cols) << endl;
        cout << OUT(type_str(descriptors.row(row_idx).type())) << endl;
        */
        aggr.push_back(em.predict_ex(descriptors.row(row_idx)));
        //em.predict_ex(descriptors.row(row_idx)).copyTo(dst);
    }

    //cout << OUT(aggr) << endl;

    Mat result;
    reduce(aggr, result, 0, CV_REDUCE_MAX);
    return result;
}

void learn_training_matrix(const Mat& descriptors,
                          const FileSet& train_pos,
                          const FileSet& train_neg,
                          Mat& train_input,
                          Mat& train_outputs) {
    int sift_clusters = 5;
    ExtEM em(sift_clusters, EM::COV_MAT_GENERIC);
    Mat log_likelihoods;
    if (!em.train(descriptors)) {
        cout << "error training" << endl;
        exit(1);
    }

    cout << OUT(em.isTrained()) << endl;
    cout << OUT(em.getMat("means")) << endl;

    train_input = Mat::zeros(train_pos.size() + train_neg.size(),
                             sift_clusters, CV_32FC1);
    train_outputs = Mat::zeros(train_pos.size() + train_neg.size(),
                               1, CV_32FC1);

    size_t res_idx = 0;
    for (size_t img_idx = 0; img_idx < train_pos.size(); ++img_idx, ++res_idx) {
        Mat img_des = load_sift_descriptors(train_pos[img_idx]);
        aggregate_predictions(em, img_des).copyTo(train_input.row(res_idx));
        train_outputs.at<float>(res_idx) = 1;
    }

    // TODO: do negative examples.
}
/*
  1. load positive training/testing sets
  2. cluster based on positive training
  3. load negative training/testing sets
  4. build training matrix based on positive/negativ training
  5. train neural net
  6. using posive/negative testing sets for testing.
 */

// objrec positive_dir negative_dir...
int main( int argc, char** argv )
{
    if (argc != 3) {
        usage();
        return 1;
    }

    FileSet pos_examples = glob_ex(std::string(argv[1]) + "/*.jpg");
    FileSet train_examples;
    FileSet test_examples;
    split_set(pos_examples, 0.66, train_examples, test_examples);

    cout << OUT(pos_examples.size()) << endl;
    cout << OUT(train_examples.size()) << endl;
    cout << OUT(train_examples) << endl;
    cout << OUT(test_examples.size()) << endl;
    cout << OUT(test_examples) << endl;

    // TESTING: maximum training size for testing peformance.
    // remove when done testing...
    train_examples = fn::slice(train_examples, 0, 50);
    Mat descriptors = load_sift_descriptors(train_examples);
    cout << OUT(descriptors.rows) << endl;
    cout << OUT(descriptors.rowRange(0, 5)) << endl;
    cout << OUT(descriptors.rowRange(descriptors.rows - 5, descriptors.rows)) << endl;


    // opencv neural networks want 32 bit floats.
    Mat converted_descs;
    descriptors.convertTo(converted_descs, CV_32FC1);

    // TODO: populate.
    FileSet neg_train_examples;
    Mat nn_train_input;
    Mat nn_train_ouput;
    learn_training_matrix(converted_descs,
                          train_examples,
                          neg_train_examples,
                          nn_train_input,
                          nn_train_ouput);

    cout << OUT(nn_train_input) << endl;
    cout << OUT(nn_train_ouput) << endl;

    return 0;
    /*
    Mat image;
    image = imread( argv[1], 1 );

    if( argc != 2 || !image.data )
    {
        printf( "No image data \n" );
        return -1;
    }
    SIFT sift;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift(image, noArray(), keypoints, descriptors, false);

    cout << OUT(descriptors.rowRange(0, 5)) << endl;
    cout << OUT(descriptors.type()) << endl;
    cout << type_str(descriptors.type()) << endl;

    ExtEM em(2);
    Mat log_likelihoods;
    if(!em.train(descriptors)) {
        cout << "error training" << endl;
        exit(1);
    }

    cout << OUT(em.isTrained()) << endl;
    cout << OUT(em.getMat("means")) << endl;

    //em.diag();
    
    std::vector<Scalar> colors = {{0,0,0,255}, {255,0,0,255}, {0,255,0,255},
                                  {0,0,255,255}, {255,255,255,255}};

    */
    /*
    cout << colors[1] << endl;
    */

    /*
    colors.push_back(Scalar(0,0,0,255));
    colors.push_back(Scalar(255,0,0,255));
    colors.push_back(Scalar(0,255,0,255));
    colors.push_back(Scalar(255,255,255,255));*/
    /*
    cout << "predict_ex results:" << endl;
    for (int ii = 0; ii < descriptors.rows; ++ii) {
        Mat ex_out = em.predict_ex(descriptors.row(ii));
    */
        /*
        Mat ex_out = em.predict_ex(descriptors.row(ii));

        Mat norm_out;
        Vec2d result = em.predict(descriptors.row(ii), norm_out);

        if (norm_out.at<double>(0) < norm_out.at<double>(1)) {
            assert(ex_out.at<double>(0) < ex_out.at<double>(1));
        }

        if (norm_out.at<double>(0) >= norm_out.at<double>(1)) {
            assert(ex_out.at<double>(0) >= ex_out.at<double>(1));
        }
        //cout << "res: " << result[0] << ", " << result[1] << endl;
        //cout << "out: " << out << endl;

        cout << OUT(ex_out) << endl;
        cout << OUT(max_idx(ex_out)) << endl;
        cout << OUT(result[1]) << endl;


        cout << OUT(colors[max_idx(ex_out)]) << endl;
        */
    /*
        drawKeypoints(image, fn::slice(keypoints, ii, ii + 1), image, colors[max_idx(ex_out)],
                      DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    */
    
    /*drawKeypoints(image, keypoints, image);*/
    /*    cout << keypoints.size() << endl;
    cout << slice(keypoints, 0, 10) << endl;
    cout << descriptors.cols << endl;*/
    
    /*
    cvtColor(image, gray_image, CV_RGB2GRAY);
    std::cout << gray_image.size() << std::endl;
    std::cout << gray_image << std::endl;*/
    /*
    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
    imshow( "Display Image", image );

    waitKey(0);

    return 0;*/
}
