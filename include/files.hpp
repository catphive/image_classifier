#ifndef FILES_HPP
#define FILES_HPP

#include <vector>
#include <string>

typedef std::vector<std::string> FileSet;

/*
  Return all files that match pattern.
 */
FileSet glob_ex(const std::string& pattern);

/*
  Randomly shuffle order of files in set.
 */
void shuffle(FileSet& set);

/*
  Randomly split in_set into train_out and test_out where
  train_out contains train_frac fraction of in_set.
 */
void split_set(FileSet in_set, double train_frac,
               FileSet& train_out, FileSet& test_out);

#endif // FILES_HPP
