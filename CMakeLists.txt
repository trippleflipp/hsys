cmake_minimum_required(VERSION 3.20)
project(hsys_dev_container LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include(cmake/deps/deps_ops.cmake)
include(cmake/deps/gtest.cmake)
include(cmake/deps/gbench.cmake)
include(cmake/deps/eigen.cmake)
include(cmake/utils/cuda_setup_targets.cmake)

enable_testing()
add_subdirectory(work1)
add_subdirectory(work2)