//
// Created by Thoh Testarossa on 2019-03-13.
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H
#define GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H

#include "../../../include/GPUconfig.h"
#include "../BellmanFordGPU.h"

#include <cuda_runtime.h>

__global__ void MSGApply_kernel(int numOfUnits, ComputeUnit<double> *computeUnits, double *mValueSet, int numOfInitV);

cudaError_t MSGApply_kernel_exec(int numOfUnits, ComputeUnit<double> *computeUnits, double *mValueSet, int numOfInitV);

__global__ void MSGGenMerge_kernel(int numOfUnits, ComputeUnit<double> *computeUnits, unsigned long long int *mValueSet,
                                   int numOfInitV);

cudaError_t MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<double> *computeUnits, unsigned long long int *mValueSet,
                                    int numOfInitV);
#endif //GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H
