//
// Created by cave-g-f on 2019-09-22
//

#pragma once

#ifndef GRAPH_ALGO_PAGERANKGPU_H
#define GRAPH_ALGO_PAGERANKGPU_H

#include "PageRank.h"
#include "../../include/GPUconfig.h"
#include "kernel_src/PageRankGPU_kernel.h"

template <typename VertexValueType, typename MessageValueType>
class PageRankGPU : public PageRank<VertexValueType, MessageValueType>
{
public:
    PageRankGPU();

    void Init(int vCount, int eCount, int numOfInitV, int computeUnitsCnt = 0) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void IterationInit(int vCount, int eCount, MessageValueType *mValues) override;
    void IterationEnd(MessageValueType *mValues) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    int MSGApply_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                       MessageValueType *mValues) override;

    int MSGGenMerge_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                          MessageValueType *mValues) override;

    //Subgraph reflection-based compression
    int reflect(const std::vector<int> &originalIntList, int originalIntRange, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);

    Graph<VertexValueType> reflectG(const Graph<VertexValueType> &o_g, const std::vector<Edge> &eSet, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);
    MessageSet<MessageValueType> reflectM(const MessageSet<MessageValueType> &o_mSet, int vCount, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);


protected:
    int vertexLimit;
    int mPerMSGSet;
    int ePerEdgeSet;

    ComputeUnit<VertexValueType> *d_computeUnits;
    MessageValueType *d_mTransformedMergedMSGValueSet;

private:

    auto MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                ComputeUnit<VertexValueType> *computeUnits);

    auto MSGApply_GPU_VVCopy(int computeUnitCount,
                             ComputeUnit<VertexValueType> *computeUnits);
};

#endif //GRAPH_ALGO_PAGERANKGPU_H
