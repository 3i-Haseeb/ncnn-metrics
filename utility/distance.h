#pragma once

#include <net.h>
#include <iostream>
#include "./typedef.h"
#include "Eigen/core"

class Distance {
 public:
  // Distance(const FEATURE& arr1, const FEATURE& arr2) : f1(arr1), f2(arr2) {}
  float euclidean_distance(const FEATURE& arr1, const FEATURE& arr2);
  float manhattan_distance(const FEATURE& arr1, const FEATURE& arr2);
  float cosine_similarity(const FEATURE& arr1, const FEATURE& arr2);
};
