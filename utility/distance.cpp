#include "./distance.h"

float Distance::euclidean_distance(const FEATURE& arr1, const FEATURE& arr2) {
  auto diff = arr1 - arr2;
  auto dist = diff.norm();
  return dist;
}

float Distance::manhattan_distance(const FEATURE& arr1, const FEATURE& arr2) {
  auto diff = arr1 - arr2;
  auto abs = diff.cwiseAbs();
  auto dist = abs.sum();
  return dist;
}

float Distance::cosine_similarity(const FEATURE& arr1, const FEATURE& arr2) {
  auto dist = arr1.dot(arr2) / (arr1.norm() * arr2.norm());
  return dist;
}
