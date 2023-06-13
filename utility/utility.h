#ifndef UTILITY_H
#define UTILITY_H

#include <net.h>
#include <iostream>

void printImage(const ncnn::Mat& mat);
void printMinMaxValues(const ncnn::Mat& mat, float* max_val, float* min_val);

#endif
