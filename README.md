# GX-Plug分布式图计算中间件
[主页](http://120.76.141.20).
目前嵌入GX-Plug中间件的PowerGraph已支持Docker容器部署。

[Docker Hub地址](https://hub.docker.com/r/kssamwang/gx-plug/tags).

- v2.0/v2.1

```sh
docker pull kssamwang/gx-plug:v2.0-PowerGraph
docker pull kssamwang/gx-plug:v2.1-PowerGraph
```

- v2.2/v2.3

由Dockerfile构建，基础镜像nvidia/cuda:11.4.0-devel-ubuntu20.04 

```sh
docker pull kssamwang/gx-plug:v2.2-PowerGraph
docker pull kssamwang/gx-plug:v2.3-PowerGraph
```

- v3.0

由Dockerfile构建，基础镜像nvidia/cuda:10.0-devel-ubuntu18.04 

完全支持GPU，包含了静态编译的三个算法的可执行文件，GX-Plug的测试使用案例，以及原版PowerGraph图分析组件链接后的可执行文件

```sh
docker pull kssamwang/gx-plug:v3.0-PowerGraph
```

推荐使用NVIDIA V100，避免其他架构出现一些兼容问题。
