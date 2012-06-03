#include "files.hpp"
#include <glob.h>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <algorithm>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <random>

using std::vector;
using std::string;
using std::cerr;
using std::endl;

vector<string> glob_ex(const string& pattern) {
    glob_t glob_buf;
    if(int err = glob(pattern.c_str(), 0, NULL, &glob_buf)) {
        throw std::runtime_error(strerror(err));
    }
    
    vector<string> result(glob_buf.gl_pathc);
    for (size_t idx = 0; idx < glob_buf.gl_pathc; ++idx) {
        result[idx] = glob_buf.gl_pathv[idx];
    }

    return result;
}

// Brendan Miller
// DATE: April 13
// TITLE: get_seed
// DESCRIPTION: returns a random seed based on time of day.
unsigned long get_seed() {
    timeval t1;
    if(gettimeofday(&t1, NULL)) {
        perror("couldn't get random seed from time");
        exit(1);
    }
    return t1.tv_usec * t1.tv_sec;
}

void shuffle(FileSet& set) {
    static std::default_random_engine engine(get_seed());
    std::shuffle(set.begin(), set.end(), engine);
}

void split_set(FileSet set, double train_frac,
               FileSet& train_out, FileSet& test_out) {
    assert(train_frac >= 0);
    shuffle(set);

    size_t train_size = set.size() * train_frac;
    train_out.insert(train_out.end(), set.begin(), set.begin() + train_size);
    test_out.insert(test_out.end(), set.begin() + train_size, set.end());
}
