//
// Created by Thoh Testarossa on 2019-03-13.
//

#pragma once

#ifndef GRAPH_ALGO_PAGERANKGPU_KERNEL_H
#define GRAPH_ALGO_PAGERANKGPU_KERNEL_H

#include "../../../include/GPUconfig.h"
#include "../PageRankGPU.h"

#include <cuda_runtime.h>

__global__ void MSGApply_kernel(Vertex *vSet, double *vValues, int numOfMsg, int *mDstSet, PRA_MSG *mValueSet, double resetProb);

cudaError_t MSGApply_kernel_exec(Vertex *vSet, double *vValues, int numOfMsg, int *mDstSet, PRA_MSG *mValueSet, double resetProb);

__global__ void MSGGenMerge_kernel(PRA_MSG *mTransformdMergedMSGValueSet, Vertex *vSet, double *vValues, int numOfEdge, Edge *eSet, int batchCnt, double threshold);

cudaError_t MSGGenMerge_kernel_exec(PRA_MSG *mTransformdMergedMSGValueSet, Vertex *vSet, double *vValues, int numOfEdge, Edge *eSet, int batchCnt, double threshold);

#endif //GRAPH_ALGO_PageRankGPU_KERNEL_H
