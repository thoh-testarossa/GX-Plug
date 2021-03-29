//
// Created by cave-g-f on 2019-09-22.
//

#include "LabelPropagationGPU.h"
#include "kernel_src/LabelPropagationGPU_kernel.h"

#include <iostream>
#include <algorithm>
#include <chrono>

#define NULLMSG -1

//Internal method for different GPU copy situations in BF algo
template<typename VertexValueType, typename MessageValueType>
auto LabelPropagationGPU<VertexValueType, MessageValueType>::MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                                                                    ComputeUnit<VertexValueType> *computeUnits)
{
    cudaError_t err = cudaMemcpy(this->d_computeUnits, computeUnits, computeUnitCount * sizeof(ComputeUnit<LPA_Value>),
                                 cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
auto
LabelPropagationGPU<VertexValueType, MessageValueType>::MSGApply_GPU_VVCopy(int computeUnitCount,
                                                                            ComputeUnit<VertexValueType> *computeUnits)
{
}

template<typename VertexValueType, typename MessageValueType>
LabelPropagationGPU<VertexValueType, MessageValueType>::LabelPropagationGPU()
{

}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagationGPU<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV,
                                                                  int computeUnitsCnt)
{
    LabelPropagation<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV, computeUnitsCnt);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagationGPU<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g,
                                                                       std::set<int> &activeVertices,
                                                                       const std::vector<int> &initVList)
{
    LabelPropagation<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagationGPU<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    LabelPropagation<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **) &d_mTransformedMergedMSGValueSet, std::max(vCount, eCount) * sizeof(MessageValueType));

    err = cudaMalloc((void **) &d_computeUnits, this->maxComputeUnits * sizeof(ComputeUnit<VertexValueType>));
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagationGPU<VertexValueType, MessageValueType>::Free()
{
    LabelPropagation<VertexValueType, MessageValueType>::Free();

    cudaFree(this->d_mTransformedMergedMSGValueSet);

    cudaFree(this->d_computeUnits);
}

template<typename VertexValueType, typename MessageValueType>
int LabelPropagationGPU<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
                                                                           ComputeUnit<VertexValueType> *computeUnits,
                                                                           MessageValueType *mValues)
{
    std::cout << "=============MSGApply_array time=============" << std::endl;

    cudaError_t err = cudaSuccess;

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < computeUnitCount; i += NUMOFGPUCORE)
    {
        int computeUnitsUsedForGPU = (computeUnitCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (computeUnitCount - i);
        err = MSGApply_kernel_exec(computeUnitsUsedForGPU, &this->d_computeUnits[i],
                                   (MessageValueType *) d_mTransformedMergedMSGValueSet);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "kernel Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    start = std::chrono::system_clock::now();
    err = cudaMemcpy(computeUnits, this->d_computeUnits, computeUnitCount * sizeof(ComputeUnit<VertexValueType>),
                     cudaMemcpyDeviceToHost);

    end = std::chrono::system_clock::now();
    std::cout << "copyback time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int
LabelPropagationGPU<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                          ComputeUnit<VertexValueType> *computeUnits,
                                                                          MessageValueType *mValues)
{
    MSGGenMerge_GPU_MVCopy(computeUnitCount, computeUnits);
    return 0;
}
