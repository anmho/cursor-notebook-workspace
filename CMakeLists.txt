cmake_minimum_required(VERSION 3.24)
project(ml_lab_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures; set explicitly on remote CI")

add_executable(vector_add kernels/vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
