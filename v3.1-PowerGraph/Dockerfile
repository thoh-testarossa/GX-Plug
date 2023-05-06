# This Dockerfile is created by kssamwang on 2023-05-06，as a test version for GX-Plug
# Tencent Cloud Server : Ubuntu 20.04 NVIDIA V100 cuda 11.4
# Docker version: 20.10.12
# Image build : docker build -t kssamwang/gx-plug:v3.1-PowerGraph .
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:11.4.0-devel-ubuntu20.04 as base

FROM base as base-amd64

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list \
    && echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update
RUN apt-get install gcc g++ build-essential libopenmpi-dev openmpi-bin default-jdk cmake zlib1g-dev git wget -y
RUN wget https://github.com/kssamwang/PowerGraphGPU/releases/download/v5.0/PowerGraphGPU-GX-Plug.tar.gz
RUN tar -zxvf PowerGraphGPU-GX-Plug.tar.gz
WORKDIR PowerGraphGPU-GX-Plug

RUN cp ./release/apps/label_propagation/lpa_graph_algo /usr/local/bin \
    && cp ./release/apps/sssp/sssp_graph_algo /usr/local/bin \
    && cp ./release/apps/pagerank/pagerank_graph_algo /usr/local/bin \
    && cp ./release/toolkits/graph_analytics/* /usr/local/bin \
    && cp ./Graph_Algo/* /usr/local/bin
