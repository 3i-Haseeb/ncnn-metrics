cmake_minimum_required(VERSION 3.17)
set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexperimental-library
# -Wno-c++98-compat -Weverything -std=c++20 -stdlib=libc++" )

project(similarityMetrics)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(ncnn REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(main main.cpp utility/utility.cpp utility/distance.cpp)

target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(main PUBLIC ${OpenCV_LIBS} ncnn Eigen3::Eigen)
