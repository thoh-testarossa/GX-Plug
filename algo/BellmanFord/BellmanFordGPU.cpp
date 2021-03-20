//
// Created by Thoh Testarossa on 2019-03-12.
//

#include "BellmanFordGPU.h"
#include "kernel_src/BellmanFordGPU_kernel.h"

#include <iostream>
#include <algorithm>
#include <chrono>

#define NULLMSG -1

//Transformation function
//This two function is the parts of MSGMerge for adapting CUDA's atomic ops
//double -> long long int
unsigned long long int doubleAsLongLongInt(double a)
{
    unsigned long long int *ptr = (unsigned long long int *) &a;
    return *ptr;
}

//long long int -> double
double longLongIntAsDouble(unsigned long long int a)
{
    double *ptr = (double *) &a;
    return *ptr;
}
//Transformation functions end

//Internal method for different GPU copy situations in BF algo
template<typename VertexValueType, typename MessageValueType>
auto BellmanFordGPU<VertexValueType, MessageValueType>::MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                                                               ComputeUnit<VertexValueType> *computeUnits)
{
    cudaError_t err = cudaMemcpy(this->d_computeUnits, computeUnits, computeUnitCount * sizeof(ComputeUnit<double>),
                                 cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
auto BellmanFordGPU<VertexValueType, MessageValueType>::MSGApply_GPU_VVCopy(int computeUnitCount,
                                                                            ComputeUnit<VertexValueType> *computeUnits)
{
    cudaError_t err = cudaMemcpy(this->d_computeUnits, computeUnits, computeUnitCount * sizeof(ComputeUnit<double>),
                                 cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
BellmanFordGPU<VertexValueType, MessageValueType>::BellmanFordGPU()
{
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFordGPU<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV, int computeUnitsCnt)
{
    BellmanFord<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV, computeUnitsCnt);
    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFordGPU<VertexValueType, MessageValueType>::IterationInit(int vCount, int eCount, MessageValueType *mValues)
{

    auto err = cudaSuccess;

    for (int i = 0; i < vCount * this->numOfInitV; i++)
        mValues[i] = doubleAsLongLongInt((double) INVALID_MASSAGE);

    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mValues,
                     vCount * this->numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFordGPU<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                             const std::vector<int> &initVList)
{
    BellmanFord<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFordGPU<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    BellmanFord<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **) &d_mTransformedMergedMSGValueSet, numOfInitV * vCount * sizeof(unsigned long long int));

    err = cudaMalloc((void **) &d_computeUnits, this->maxComputeUnits * sizeof(ComputeUnit<double>));
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFordGPU<VertexValueType, MessageValueType>::Free()
{
    BellmanFord<VertexValueType, MessageValueType>::Free();

    cudaFree(this->d_mTransformedMergedMSGValueSet);

    cudaFree(this->d_computeUnits);
}

template<typename VertexValueType, typename MessageValueType>
int BellmanFordGPU<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
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
                                   (double *) d_mTransformedMergedMSGValueSet,
                                   this->numOfInitV);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "kernel Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    start = std::chrono::system_clock::now();
    err = cudaMemcpy(computeUnits, this->d_computeUnits, computeUnitCount * sizeof(ComputeUnit<double>),
                     cudaMemcpyDeviceToHost);

    end = std::chrono::system_clock::now();
    std::cout << "copyback time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int BellmanFordGPU<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                         ComputeUnit<VertexValueType> *computeUnits,
                                                                         MessageValueType *mValues)
{
    std::cout << "=============MSGGenMerge_array time=============" << std::endl;
    //Generate merged msgs directly
    cudaError_t err = cudaSuccess;
    MSGGenMerge_GPU_MVCopy(computeUnitCount, computeUnits);

    for (int i = 0; i < computeUnitCount; i += NUMOFGPUCORE)
    {
        int computeUnitsUsedForGPU = (computeUnitCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (computeUnitCount - i);
        err = MSGGenMerge_kernel_exec(computeUnitsUsedForGPU, &this->d_computeUnits[i],
                                      this->d_mTransformedMergedMSGValueSet, this->numOfInitV);
    }

    return 0;
}
