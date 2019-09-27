cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_LabelPropagation
        LabelPropagation.h
        LabelPropagation.cpp
        LabelPropagation_impl.cpp)

target_link_libraries(algo_LabelPropagation
        core_Graph
        core_GraphUtil
        core_MessageSet)