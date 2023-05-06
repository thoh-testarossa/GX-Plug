# This Dockerfile is created by kssamwang on 2023-05-06，as a test version for GX-Plug
# Tencent Cloud Server : Ubuntu 20.04 NVIDIA V100 cuda 11.4
# Docker version: 20.10.12
# Image build : docker build -t kssamwang/gx-plug:v3.0-GraphX .
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

WORKDIR /root
RUN apt-get update
RUN apt-get install gcc g++ build-essential libopenmpi-dev openmpi-bin default-jdk cmake zlib1g-dev git wget -y
RUN wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.deb && dpkg -i scala-2.11.12.deb
RUN git clone https://github.com/kssamwang/Spark-GraphX.git ./Spark-GraphX
WORKDIR ./Spark-GraphX
RUN build/mvn -DskipTests clean package && build/mvn install
WORKDIR /root
RUN wget https://github.com/kssamwang/GraphXGPU/releases/download/v3.0/GraphXGPU-gxplug-v3.0.tar.gz \
    && tar xzvf GraphXGPU-gxplug-v3.0.tar.gz
RUN mv ./build /GraphX

