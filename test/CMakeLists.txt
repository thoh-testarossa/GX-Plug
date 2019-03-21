cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_executable(algo_BellmanFordTest
        BellmanFordTest.cpp)

target_link_libraries(algo_BellmanFordTest
        algo_BellmanFord
        core_Graph
        core_MessageSet
        core_GraphUtil)

add_executable(core_GraphAccessTest
        GraphAccessTest.cpp)

target_link_libraries(core_GraphAccessTest
        core_Graph
        core_MessageSet
        core_GraphUtil)

if(CUDA_FOUND)
    add_executable(algo_BellmanFordGPUTest
            BellmanFordGPUTest.cpp)

    target_link_libraries(algo_BellmanFordGPUTest
            algo_BellmanFordGPU
            core_Graph
            core_MessageSet
            core_GraphUtil)
endif(CUDA_FOUND)

