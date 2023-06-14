#include "./utility.h"

// Convert ncnn to eigen matric
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

void printImage(const ncnn::Mat& mat) {
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

void getMinMaxValues(const ncnn::Mat& mat, float* max_val, float* min_val) {
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
