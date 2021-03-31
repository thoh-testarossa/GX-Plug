//
// Created by cave-g-f on 2019-10-11.
//

#pragma once

#ifndef GRAPH_ALGO_LABELPROPAGATIONGPU_KERNEL_H
#define GRAPH_ALGO_LABELPROPAGATIONGPU_KERNEL_H

#include "../../../include/GPUconfig.h"
#include "../LabelPropagationGPU.h"

#include <cuda_runtime.h>

__global__ void MSGApply_kernel(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet);

cudaError_t MSGApply_kernel_exec(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet);

__global__ void
MSGGenMerge_kernel(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet, int msgIndex);

cudaError_t
MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet, int msgIndex);

#endif //GRAPH_ALGO_LABELPROPAGATIONGPU_KERNEL_H
