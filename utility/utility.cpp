#include "utility.h"

void printImage(const ncnn::Mat &mat) {
  int width = mat.w;
  int height = mat.h;
  int channels = mat.c;

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        float value = mat.channel(c).row(h)[w];
        std::cout << value << " ";
      }
    }
    std::cout << std::endl;
  }
}

void printMinMaxValues(const ncnn::Mat &mat, float *max_val, float *min_val) {
  int size = mat.total();
  int width = mat.w;
  int height = mat.h;
  int channels = mat.c;

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        float value = mat.channel(c).row(h)[w];
        *max_val = std::max(*max_val, value);
        *min_val = std::min(*min_val, value);
      }
    }
  }
}
