#pragma once

#include <net.h>
#include <iostream>
#include "Eigen/core"

auto static constexpr FEATURE_SIZE = int{512};
typedef Eigen::Matrix<float, 1, FEATURE_SIZE, Eigen::RowMajor>
    FEATURE;  //[1,512]

class Distance {
 public:
  Distance(const FEATURE& arr1, const FEATURE& arr2);
};

FEATURE ncnn_to_eigen(const ncnn::Mat& mat);
float euclidean_distance(const FEATURE& arr1, const FEATURE& arr2);
float manhattan_distance(const FEATURE& arr1, const FEATURE& arr2);
float cosine_similarity(const FEATURE& arr1, const FEATURE& arr2);
