# This Dockerfile is created by kssamwang on 2023-04-18，as a test version for GX-Plug
# Tencent Cloud Server : Ubuntu 20.04 NVIDIA V100 cuda 11.4
# Docker version: 20.10.12
# Image build : docker build -t kssamwang/gx-plug:v2.2-PowerGraph .
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:11.4.0-devel-ubuntu20.04 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 8.2.4.15
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.4"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.4"

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo  "deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update
RUN apt-get install cmake=3.16.3-1ubuntu1.20.04.1 -y
RUN apt-get install openssl -y
RUN apt-get install git -y
WORKDIR /root/GX-Plug
RUN git clone https://github.com/thoh-testarossa/GX-Plug.git -b powergraph-master /root/GX-Plug
WORKDIR /root/GX-Plug/build
RUN cd /root/GX-Plug/build
RUN cmake ..
RUN make -j8

RUN cp -f /root/GX-Plug/build/bin/algo_BellmanFordGPUTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/algo_PageRankTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_BellmanFord /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_PageRankGPU /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/algo_BellmanFordTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/core_GraphUtilTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_BellmanFordGPU /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/tool_RandGraphGen /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/algo_LabelPropagationGPUTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilClientTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_LabelPropagation /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/tool_TwitterGraphProcessing /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/algo_LabelPropagationTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilClientTest_LabelPropagation /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_LabelPropagationGPU  /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/tool_UK2007GraphProcessing /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/algo_PageRankGPUTest /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilClientTest_PageRank /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/srv_UtilServerTest_PageRank /usr/local/bin
RUN cp -f /root/GX-Plug/build/bin/tool_WRNGraphProcessing /usr/local/bin

