cmake_minimum_required(VERSION 3.22)

project(dot_product LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python bindings
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

nanobind_add_module(fast_dot fast_dot.cu)
target_include_directories(fast_dot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fast_dot PRIVATE -lineinfo)  # Useful for profiling in Nsight
target_compile_options(fast_dot PRIVATE -use_fast_math)  # Faster math functions

nanobind_add_module(gxd gxd.cu)
target_include_directories(gxd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gxd PRIVATE -lineinfo)  # Useful for profiling in Nsight
target_compile_options(gxd PRIVATE -use_fast_math)  # Faster math functions