cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

add_library(core_Graph
        Graph.h
        Graph.cpp
        Graph_impl.cpp)

add_library(core_GraphUtil
        GraphUtil.h
        GraphUtil.cpp
        GraphUtil_impl.cpp)

add_library(core_MessageSet
        MessageSet.h
        MessageSet.cpp
        MessageSet_impl.cpp)
