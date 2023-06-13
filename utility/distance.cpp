#include "./distance.h"
// #include <net.h>
// #include <iostream>
// #include "Eigen/core"

FEATURE ncnn_to_eigen(const ncnn::Mat& mat) {
  int rows = mat.h;
  int cols = mat.w;
  const float* data = mat.channel(0);

  // Copy the data from the ncnn::Mat to the Eigen matrix
  Eigen::MatrixXf eigenMat(rows, cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      eigenMat(row, col) = data[row * cols + col];
    }
  }

  return eigenMat;
}

float euclidean_distance(const FEATURE& arr1, const FEATURE& arr2) {
  auto diff = arr1 - arr2;
  auto dist = diff.norm();
  return dist;
}

float manhattan_distance(const FEATURE& arr1, const FEATURE& arr2) {
  auto diff = arr1 - arr2;
  auto abs = diff.cwiseAbs();
  auto dist = abs.sum();
  return dist;
}

float cosine_similarity(const FEATURE& arr1, const FEATURE& arr2) {
  auto dist = arr1.dot(arr2) / (arr1.norm() * arr2.norm());
  return dist;
}
