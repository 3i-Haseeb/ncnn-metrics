#pragma once
#include "Eigen/core"

auto static constexpr FEATURE_SIZE = int{512};

typedef Eigen::Matrix<float, 1, FEATURE_SIZE, Eigen::RowMajor>
    FEATURE;  //[1,512]
