#pragma once

#include <net.h>
#include <iostream>
#include "./typedef.h"
#include "Eigen/core"

void printImage(const ncnn::Mat& mat);
void printMinMaxValues(const ncnn::Mat& mat, float* max_val, float* min_val);
FEATURE ncnn_to_eigen(const ncnn::Mat& mat);
