# GX-Plug分布式图计算中间件
[主页](http://120.76.141.20).
目前嵌入GX-Plug中间件的PowerGraph已支持Docker容器部署
[Docker Hub地址](https://hub.docker.com/r/kssamwang/gx-plug/tags).

v2.0/v2.1基于手动构建
v2.2+由Dockerfile构建，基础镜像nvidia/cuda:11.4.0-devel-ubuntu20.04 
```sh
docker pull kssamwang/gx-plug:v2.2-PowerGraph
docker pull kssamwang/gx-plug:v2.3-PowerGraph
```