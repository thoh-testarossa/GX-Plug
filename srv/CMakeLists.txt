cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

add_library(srv_UNIX_shm
        UNIX_shm.h
        UNIX_shm.cpp)

add_library(srv_UNIX_msg
        UNIX_msg.h
        UNIX_msg.cpp)