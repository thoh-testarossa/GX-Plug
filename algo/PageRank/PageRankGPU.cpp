//
// Created by cave-g-f on 2019-09-22.
//

#include "PageRankGPU.h"
#include "kernel_src/PageRankGPU_kernel.h"

#include <iostream>
#include <algorithm>
#include <chrono>

#define NULLMSG -1

//Internal method for different GPU copy situations in BF algo
template<typename VertexValueType, typename MessageValueType>
auto PageRankGPU<VertexValueType, MessageValueType>::MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                                                            ComputeUnit<VertexValueType> *computeUnits)
{
    cudaError_t err = cudaMemcpy(this->d_computeUnits, computeUnits,
                                 computeUnitCount * sizeof(ComputeUnit<VertexValueType>),
                                 cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
auto PageRankGPU<VertexValueType, MessageValueType>::MSGApply_GPU_VVCopy(int computeUnitCount,
                                                                         ComputeUnit<VertexValueType> *computeUnits)
{
    cudaError_t err = cudaMemcpy(this->d_computeUnits, computeUnits,
                                 computeUnitCount * sizeof(ComputeUnit<VertexValueType>),
                                 cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
PageRankGPU<VertexValueType, MessageValueType>::PageRankGPU()
{

}

template<typename VertexValueType, typename MessageValueType>
void PageRankGPU<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV, int computeUnitsCnt)
{
    PageRank<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV, computeUnitsCnt);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;
}

template<typename VertexValueType, typename MessageValueType>
void
PageRankGPU<VertexValueType, MessageValueType>::IterationInit(int vCount, int eCount, MessageValueType *mValues)
{

    auto err = cudaSuccess;

    for (int i = 0; i < vCount; i++)
        mValues[i] = MessageValueType(-1, 0);

    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mValues, vCount * sizeof(MessageValueType),
                     cudaMemcpyHostToDevice);
}

template<typename VertexValueType, typename MessageValueType>
void PageRankGPU<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                               const std::vector<int> &initVList)
{
    PageRank<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template<typename VertexValueType, typename MessageValueType>
void PageRankGPU<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    PageRank<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **) &d_mTransformedMergedMSGValueSet, vCount * sizeof(MessageValueType));

    err = cudaMalloc((void **) &d_computeUnits, this->maxComputeUnits * sizeof(ComputeUnit<VertexValueType>));
}

template<typename VertexValueType, typename MessageValueType>
void PageRankGPU<VertexValueType, MessageValueType>::Free()
{
    PageRank<VertexValueType, MessageValueType>::Free();

    cudaFree(this->d_mTransformedMergedMSGValueSet);

    cudaFree(this->d_computeUnits);
}

template<typename VertexValueType, typename MessageValueType>
int PageRankGPU<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
                                                                   ComputeUnit<VertexValueType> *computeUnits,
                                                                   MessageValueType *mValues)
{
//    std::cout << "=============MSGApply_array time=============" << std::endl;

    cudaError_t err = cudaSuccess;

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < computeUnitCount; i += NUMOFGPUCORE)
    {
        int computeUnitsUsedForGPU = (computeUnitCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (computeUnitCount - i);
        err = MSGApply_kernel_exec(computeUnitsUsedForGPU, &this->d_computeUnits[i], d_mTransformedMergedMSGValueSet);
    }
    auto end = std::chrono::system_clock::now();
//    std::cout << "kernel Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
//              << std::endl;

    start = std::chrono::system_clock::now();
    err = cudaMemcpy(computeUnits, this->d_computeUnits, computeUnitCount * sizeof(ComputeUnit<VertexValueType>),
                     cudaMemcpyDeviceToHost);

    end = std::chrono::system_clock::now();
//    std::cout << "copyback time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
//              << std::endl;

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int PageRankGPU<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                      ComputeUnit<VertexValueType> *computeUnits,
                                                                      MessageValueType *mValues)
{
//    std::cout << "=============MSGGenMerge_array time=============" << std::endl;
    //Generate merged msgs directly
    cudaError_t err = cudaSuccess;
    MSGGenMerge_GPU_MVCopy(computeUnitCount, computeUnits);

    for (int i = 0; i < computeUnitCount; i += NUMOFGPUCORE)
    {
        int computeUnitsUsedForGPU = (computeUnitCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (computeUnitCount - i);
        err = MSGGenMerge_kernel_exec(computeUnitsUsedForGPU, &this->d_computeUnits[i],
                                      this->d_mTransformedMergedMSGValueSet, this->resetProb, this->deltaThreshold);
    }

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int
PageRankGPU<VertexValueType, MessageValueType>::reflect(const std::vector<int> &originalIntList, int originalIntRange,
                                                        std::vector<int> &reflectIndex, std::vector<int> &reversedIndex)
{
    //Reflection
    int reflectCount = 0;

    for (auto o_i : originalIntList)
    {
        if (reversedIndex.at(o_i) == NO_REFLECTION)
        {
            reflectIndex.emplace_back(o_i);
            reversedIndex.at(o_i) = reflectCount++;
        }
    }

    return reflectCount;
}

template<typename VertexValueType, typename MessageValueType>
Graph<VertexValueType> PageRankGPU<VertexValueType, MessageValueType>::reflectG(const Graph<VertexValueType> &o_g,
                                                                                const std::vector<Edge> &eSet,
                                                                                std::vector<int> &reflectIndex,
                                                                                std::vector<int> &reversedIndex)
{
    //Init
    int vCount = o_g.vCount;
    int eCount = eSet.size();

    reflectIndex.clear();
    reversedIndex.clear();
    reflectIndex.reserve(2 * eCount);
    reversedIndex.reserve(vCount);
    reversedIndex.assign(vCount, NO_REFLECTION);

    //Calculate reflection using eSet and generate reflected eSet
    auto r_eSet = std::vector<Edge>();
    r_eSet.reserve(2 * eCount);

    auto originalIntList = std::vector<int>();
    originalIntList.reserve(2 * eCount);

    for (const auto &e : eSet)
    {
        originalIntList.emplace_back(e.src);
        originalIntList.emplace_back(e.dst);
    }

    int reflectCount = this->reflect(originalIntList, vCount, reflectIndex, reversedIndex);

    //Generate reflected eSet
    for (const auto &e : eSet)
        r_eSet.emplace_back(reversedIndex.at(e.src), reversedIndex.at(e.dst), e.weight);

    //Generate reflected vSet & vValueSet
    auto r_vSet = std::vector<Vertex>();
    r_vSet.reserve(reflectCount * sizeof(Vertex));

    auto r_vValueSet = std::vector<VertexValueType>();
    r_vValueSet.reserve(reflectCount * sizeof(VertexValueType));

    for (int i = 0; i < reflectCount; i++)
    {
        r_vSet.emplace_back(o_g.vList.at(reflectIndex.at(i)));
        r_vValueSet.emplace_back(o_g.verticesValue.at(reflectIndex.at(i)));
        r_vSet.at(i).vertexID = i;
    }

    //Generate reflected graph and return
    return Graph<VertexValueType>(r_vSet, r_eSet, r_vValueSet);
}

template<typename VertexValueType, typename MessageValueType>
MessageSet<MessageValueType>
PageRankGPU<VertexValueType, MessageValueType>::reflectM(const MessageSet<MessageValueType> &o_mSet, int vCount,
                                                         std::vector<int> &reflectIndex,
                                                         std::vector<int> &reversedIndex)
{
    auto r_mSet = MessageSet<MessageValueType>();

    reflectIndex.reserve(o_mSet.mSet.size());
    reversedIndex.reserve(vCount);
    reversedIndex.assign(vCount, NO_REFLECTION);

    auto originalIntList = std::vector<int>();
    originalIntList.reserve(o_mSet.mSet.size());

    for (const auto &m : o_mSet.mSet) originalIntList.emplace_back(m.dst);

    int reflectCount = this->reflect(originalIntList, vCount, reflectIndex, reversedIndex);

    for (auto &m : o_mSet.mSet)
    {
        r_mSet.insertMsg(Message<MessageValueType>(m.src, reversedIndex.at(m.dst), m.value));
    }

    return r_mSet;
}



