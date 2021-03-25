//
// Created by cave-g-f on 2019-09-22.
//

#pragma once

#ifndef GRAPH_ALGO_PAGERANKGPU_KERNEL_H
#define GRAPH_ALGO_PAGERANKGPU_KERNEL_H

#include "../../../include/GPUconfig.h"
#include "../PageRankGPU.h"

#include <cuda_runtime.h>

__global__ void
MSGApply_kernel(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet);

cudaError_t
MSGApply_kernel_exec(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet);

__global__ void
MSGGenMerge_kernel(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet,
                   double resetProb, double deltaThreshold);

cudaError_t
MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet,
                        double resetProb, double deltaThreshold);

#endif //GRAPH_ALGO_PageRankGPU_KERNEL_H
