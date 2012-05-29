#ifndef UTIL_HPP
#define UTIL_HPP

#include <utility>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <ostream>
#include <vector>
#include <limits>
//#include <type_traits>
#include <cstdarg>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>

namespace fn {

    
    template <typename T, typename Function>
    bool all(const std::vector<T>& vec, Function func) {
        return std::all_of(vec.begin(), vec.end(), func);
    }

    
    template <typename T, typename BinaryOperation,
              typename U = typename std::result_of<BinaryOperation(const T&, const T&)>::type>
    std::vector<U> zip(const std::vector<T>& vec1, const std::vector<T>& vec2,
                       BinaryOperation func) {
        assert(vec1.size() == vec2.size());
        std::vector<U> result;
        result.reserve(vec1.size());
        transform(vec1.begin(), vec1.end(), vec2.begin(),
                  std::back_inserter(result), func);
        return result;
    }

    template <class Generator,
              typename T = typename std::result_of<Generator()>::type>
    std::vector<T> gen_n(size_t count, Generator generator) {
        std::vector<T> result;
        result.reserve(count);
        generate_n(std::back_insert_iterator<std::vector<T> >(result), count, generator);
        return result;
    }


    // python style index based slice.
    template <typename T>
    std::vector<T> slice(const std::vector<T> vec,
                         size_t start,
                         size_t end = std::numeric_limits<size_t>::max()) {
        start = std::min(start, vec.size());
        start = std::max(start, size_t(0));
        end = std::min(end, vec.size());
        end = std::max(start, end);
        return std::vector<T>(vec.begin() + start, vec.begin() + end);
    }

    template <typename T>
    T min_elem(const std::vector<T>& vec) {
        assert(vec.size());
        return *std::min_element(vec.begin(), vec.end());
    }

    template <typename T>
    T max_elem(const std::vector<T>& vec) {
        assert(vec.size());
        return *std::max_element(vec.begin(), vec.end());
    }

    template <typename T>
    size_t min_index(const std::vector<T>& vec) {
        assert(vec.size());
        return std::min_element(vec.begin(), vec.end()) - vec.begin();
    }

    template <typename T>
    size_t max_index(const std::vector<T>& vec) {
        assert(vec.size());
        return std::max_element(vec.begin(), vec.end()) - vec.begin();
    }

    template <typename T>
    T sum(const std::vector<T>& vec, T init = T()) {
        return std::accumulate(vec.begin(), vec.end(), init);
    }


    template <typename T, typename Function,
              typename ResultType = std::vector<typename std::result_of<Function(const T&)>::type> >
    ResultType vector_comprehension(const std::vector<T>& in_vec, Function func) {
        ResultType result;
        result.reserve(in_vec.size());
        for (auto& elem : in_vec) {
            result.push_back(func(elem));
        }
        return result;
    }

    template <typename T, typename Function, typename Predicate,
              typename ResultType = std::vector<typename std::result_of<Function(const T&)>::type> >
    ResultType vector_comprehension_if(const std::vector<T>& in_vec,
                                       Function func,
                                       Predicate pred) {
        ResultType result;
        result.reserve(in_vec.size());
        for (auto& elem : in_vec) {
            if (pred(elem)) {
                result.push_back(func(elem));
            }
        }
        return result;
    }

    template <typename T>
    struct Get: T {};

// python style list comprehension.
#define VFOR(coll, var, expr)                                           \
    fn::vector_comprehension                                            \
    ((coll), [&](typename fn::Get<typename std::remove_reference<decltype(coll)>::type>::const_reference var) { return (expr);})

#define VFOR_IF(coll, var, res_expr, cond_expr)                         \
    fn::vector_comprehension_if                                         \
    ((coll), [&](typename fn::Get<typename std::remove_reference<decltype(coll)>::type>::const_reference var) { return (res_expr);}, \
     [&](typename fn::Get<typename std::remove_reference<decltype(coll)>::type>::const_reference var) { return (cond_expr);})

    template <typename T>
    std::vector<T> vlit(const std::initializer_list<T>& init) {
        return std::vector<T>(init);
    }

    std::string format(const char* fmt, ...)
        __attribute__ ((format (printf, 1, 2)));

    inline std::string format(const char* fmt, ...)
    {
        std::vector<char> buf(1024);
        va_list var_args;
        va_start(var_args, fmt);
        vsnprintf(&buf[0], buf.size(), fmt, var_args);
        va_end(var_args);
        return std::string(&buf[0]);
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    stream << "[";

    typename std::vector<T>::const_iterator iter = vec.begin();
    if (iter != vec.end()) {
        stream << *iter;
        ++iter;
        for (; iter != vec.end(); ++iter) {
            stream << ", ";
            stream << *iter;
        }
    }

    stream << "]";
    return stream;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, const std::pair<T, U> pair) {
    return stream << "(" << pair.first << ", " << pair.second << ")";
}

// Quickly print a value labeled with its expression.
// Example:
// int var = 6;
// std::cout << OUT(var) << end;
// Output:
// var: 6
#define OUT(var) " " #var ": " << (var)

#define STR_CASE(expr) case expr: return #expr

#endif // UTIL_HPP
