cmake_minimum_required(VERSION 3.10)
project(cuda-nlmeans)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)
if (CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
else()
    add_definitions(-DNO_CUDA)
endif()

add_executable(cuda-nlmeans main.cpp utilities.cpp utilities.h)
target_link_libraries(cuda-nlmeans PRIVATE ${OpenCV_LIBRARIES})
target_link_libraries(cuda-nlmeans PRIVATE OpenMP::OpenMP_CXX)