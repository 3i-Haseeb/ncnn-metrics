#pragma once

#include <net.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "./typedef.h"
#include "Eigen/core"

void printImage(const ncnn::Mat& mat);
void getMinMaxValues(const ncnn::Mat& mat, float* max_val, float* min_val);
void normalize_image(cv::Mat& image);
FEATURE ncnn_to_eigen(const ncnn::Mat& mat);
void normalize_image2(cv::Mat& image,
                      const std::vector<float>& mean,
                      const std::vector<float>& std);
