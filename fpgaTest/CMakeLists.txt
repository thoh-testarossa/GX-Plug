cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

add_executable(sssp_fpga
        main_sssp.cpp)

target_link_libraries(sssp_fpga
        algo_BellmanFord
        core_Graph
        core_GraphUtil
        algo_BellmanFordFPGA
)