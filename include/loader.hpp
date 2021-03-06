#ifndef LOADER_HPP
#define LOADER_HPP

// These classes are responsible for loading descriptors of various types.

#include "files.hpp"

#include <opencv/cv.h>
#include <vector>
#include <string>

/*
  Responsible for loading descriptors from files.
 */
struct Loader {
    Loader() {}
    virtual ~Loader() {}

    /*
      Returns descriptors from image_name.
     */
    virtual cv::Mat load(const std::string& image_name) = 0;
    /*
      Returns descriptors from all images in image_set.
     */
    virtual cv::Mat load_set(const FileSet& image_set);


    /*
      Returns the appropriate number of clusters in a guassian mixture
      model of the descriptors returned by this loader.
     */
    virtual int em_clusters() = 0;
    
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
};

struct SIFTLoader: Loader {
    virtual cv::Mat load(const std::string& image_name);
    virtual int em_clusters();
};

struct ColorLoader: Loader {
    virtual cv::Mat load(const std::string& image_name);
    virtual int em_clusters();
};

typedef cv::Ptr<Loader> LoaderPtr;
typedef std::vector<LoaderPtr> LoaderVec;
LoaderVec make_loaders();

#endif // LOADER_HPP
