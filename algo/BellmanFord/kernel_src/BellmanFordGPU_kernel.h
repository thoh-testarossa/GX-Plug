//
// Created by Thoh Testarossa on 2019-03-13.
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H
#define GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H

#include "../../../include/GPUconfig.h"

#include <cuda_runtime.h>

__global__ void MSGApply_kernel(int numOfInitV, int *initVSet, double *vValues,
                                int numOfMsg, int *mDstSet, int *mInitVSet, double *mValueSet,
                                bool *AVCheckSet);

cudaError_t MSGApply_kernel_exec(int numOfInitV, int *initVSet, double *vValues,
                                 int numOfMsg, int *mDstSet, int *mInitVSet, double *mValueSet,
                                 bool *AVCheckSet);

__global__ void MSGGen_kernel(int numOfEdge, int numOfAV, int *activeVerticeSet,
                              int *eSrcSet, int *eDstSet, double *eWeightSet,
                              int numOfInitV, int *initVSet, double *vValues,
                              int *mDstSet, int *mInitVSet, double *mValueSet);

cudaError_t MSGGen_kernel_exec(int numOfEdge, int numOfAV, int *activeVerticeSet,
                               int *eSrcSet, int *eDstSet, double *eWeightSet,
                               int numOfInitV, int *initVSet, double *vValues,
                               int *mDstSet, int *mInitVSet, double *mValueSet);

__global__ void MSGMerge_kernel(unsigned long long int *mTransFormdMergedMSGValueSet,
	                            int numOfInitV, int *initVSet, 
	                            int numOfMsg, int *mDstSet, int *mInitVSet, unsigned long long int *mValueTSet);

cudaError_t MSGMerge_kernel_exec(unsigned long long int *mTransFormdMergedMSGValueSet,
	                             int numOfInitV, int *initVSet, 
	                             int numOfMsg, int *mDstSet, int *mInitVSet, unsigned long long int *mValueTSet);
#endif //GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H
