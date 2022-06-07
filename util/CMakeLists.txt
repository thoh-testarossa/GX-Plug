cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)

if(OpenCL_FOUND)
    add_library(xcl2
        ./xcl2/xcl2.cpp
        ./xcl2/xcl2.hpp)

    target_link_libraries(
        xcl2
        ${OpenCL_LIBRARIES}
    )
endif(OpenCL_FOUND)