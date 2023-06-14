// #include <algorithm>
#include <stdio.h>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <net.h>
#include <omp.h>
#include "utility/distance.h"
#include "utility/utility.h"

#include "Eigen/core"

int main(int argc, char** argv) {
  auto imagePaths = std::vector<std::string>{
      "../test-images/1.jpeg", "../test-images/2.jpeg", "../test-images/3.jpeg",
      "../test-images/4.jpeg", "../test-images/5.jpeg", "../test-images/6.jpeg",
  };
  auto results = std::vector<FEATURE>{};

  // if (argc > 0) {
  //   auto path1 = std::string{argv[1]};
  //   auto path2 = std::string{argv[2]};
  //   imagePaths.push_back("../test-images/" + path1);
  //   imagePaths.push_back("../test-images/" + path2);
  // }
  // Load NCNN model
  ncnn::Net net;
  int ret = net.load_param("./models/model_opt.param");
  if (ret)
    std::cerr << "Failed to load model parameters!" << std::endl;
  ret = net.load_model("./models/model_opt.bin");
  if (ret)
    std::cerr << "Failed to load model weights!" << std::endl;

  for (const auto& path : imagePaths) {
    // Load image
    cv::Mat orig_img = cv::imread(path, cv::IMREAD_COLOR);
    if (orig_img.empty()) {
      std::cerr << "Unable to read image file " << path << std::endl;
      return -1;
    }

    auto img = cv::Mat();
    cv::resize(orig_img, img, cv::Size(128, 256));

    // Convert image data to ncnn format
    // Opencv image is bgr, model also expects bgr
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR,
                                             img.cols, img.rows);

    // Preprocess the image
    float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f,
                          1 / .225f / 255.f};

    input.substract_mean_normalize(mean_vals, norm_vals);

    // float max_val, min_val = {};
    // getMinMaxValues(input, &max_val, &min_val);
    // std::cout << max_val << ", " << min_val << std::endl;

    // Inference
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(6);
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);

    // Flatten
    ncnn::Mat outFlattened = output.reshape(output.w * output.h * output.c);
    // std::cout << outFlattened.h;

    // // Convert to Eigen matrix
    auto features = ncnn_to_eigen(outFlattened);
    // std::cout << outEigen.rows() << std::endl;

    results.push_back(features);
  }

  auto distance = Distance{};

  for (const auto& result : results) {
    auto dist = distance.cosine_similarity(results[4], result);
    std::cout << "Distance: " << dist << std::endl;
  }
}
