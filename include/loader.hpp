#ifndef LOADER_HPP
#define LOADER_HPP

// These classes are responsible for loading descriptors of various types.

#include "files.hpp"

#include <opencv/cv.h>
#include <vector>
#include <string>

struct Loader {
    Loader() {}
    virtual ~Loader() {}

    virtual cv::Mat load(const std::string& image_name) = 0;
    virtual cv::Mat load_set(const FileSet& image_set);
    
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
};

struct SIFTLoader: Loader {
    virtual cv::Mat load(const std::string& image_name);
};

struct ColorLoader: Loader {
    virtual cv::Mat load(const std::string& image_name);
};

typedef cv::Ptr<Loader> LoaderPtr;
typedef std::vector<LoaderPtr> LoaderVec;
LoaderVec make_loaders();

#endif // LOADER_HPP
