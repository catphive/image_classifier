#include "loader.hpp"
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;

LoaderVec make_loaders() {
    return {
        {LoaderPtr(new SIFTLoader())},
        {LoaderPtr(new ColorLoader())}
    };/*
    return {
        {LoaderPtr(new SIFTLoader())},
        {LoaderPtr(new ColorLoader())}
        };*/
}

Mat ColorLoader::load(const string& image_name) {
    Mat image = imread(image_name, 1);

    /*
    cout << OUT(type_str(image.type())) << endl;
    
    namedWindow( "Base Image", CV_WINDOW_AUTOSIZE );
    imshow("Base Image", image );
    */

    Mat lab_image;
    cvtColor(image, lab_image, CV_BGR2Lab);
    //cout << OUT(type_str(lab_image.type())) << endl;

    Mat pixels(lab_image.rows * lab_image.cols, 3, CV_32FC1);
    //cout << OUT(lab_image.rows) << endl;
    //cout << OUT(lab_image.cols) << endl;
    int pixels_idx = 0;
    for(int ii = 0; ii < lab_image.rows; ++ii) {
        for(int jj = 0; jj < lab_image.cols; ++jj, ++pixels_idx) {
            Vec3b pixel = lab_image.at<Vec3b>(ii, jj);
            pixels.at<float>(pixels_idx, 0) = pixel[0];
            pixels.at<float>(pixels_idx, 1) = pixel[1];
            pixels.at<float>(pixels_idx, 2) = pixel[2];
        }
    }
    
    Mat labels;
    int nclusters = 10;
    Mat means;
    kmeans(pixels, nclusters, labels, TermCriteria(TermCriteria::EPS,
                                                   /*ignored*/50,
                                                   1.5),
                        1, KMEANS_PP_CENTERS, means);

    // DEBUG:

    //cout << OUT(labels.rowRange(200000,200050)) << endl;
    //cout << OUT(type_str(labels.type())) << endl;

    /*
    pixels_idx = 0;
    for(int ii = 0; ii < lab_image.rows; ++ii) {
        for(int jj = 0; jj < lab_image.cols; ++jj, ++pixels_idx) {
            int32_t label = labels.at<int32_t>(pixels_idx);
            Vec3b pixel(means.at<float>(label, 0),
                        means.at<float>(label, 1),
                        means.at<float>(label, 2));
            lab_image.at<Vec3b>(ii, jj) = pixel;
        }
    }

    Mat tmp_image;
    cvtColor(lab_image, tmp_image, CV_Lab2BGR);

    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
    imshow( "Display Image", tmp_image );
    waitKey(0);
    */

    return means;
}
/*
Mat ColorLoader::load_set(const FileSet& images) {
    Mat accum;
    for (const std::string& image_name : images) {
        Mat descriptors = load(image_name);

        if (accum.empty()) {
            accum = descriptors;
        } else {
            Mat tmp;
            vconcat(accum, descriptors, tmp);
            accum = tmp;
        }
    }

    return accum;
    }*/

namespace {
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
/*
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
*/
}

Mat SIFTLoader::load(const string& image_name) {
    std::vector<KeyPoint> keypoints;
    Mat image;

    return load_sift_descriptors(image_name, keypoints, image);
}

Mat Loader::load_set(const FileSet& images) {
    Mat accum;
    for (const std::string& image_name : images) {
        Mat descriptors = load(image_name);

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
