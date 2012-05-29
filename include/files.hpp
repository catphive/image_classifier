#ifndef FILES_HPP
#define FILES_HPP

#include <vector>
#include <string>

typedef std::vector<std::string> FileSet;

FileSet glob_ex(const std::string& pattern);

void split_set(FileSet in_set, double train_frac,
               FileSet& train_out, FileSet& test_out);

#endif // FILES_HPP
