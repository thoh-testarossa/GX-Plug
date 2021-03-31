//
// Created by Thoh Testarossa on 2019-03-12
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_H
#define GRAPH_ALGO_BELLMANFORDGPU_H

#include "BellmanFord.h"
#include "../../include/GPUconfig.h"

template<typename VertexValueType, typename MessageValueType>
class BellmanFordGPU : public BellmanFord<VertexValueType, MessageValueType>
{
public:
    BellmanFordGPU();

    void Init(int vCount, int eCount, int numOfInitV, int computeUnitsCnt = 0) override;

    void
    GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;

    void IterationInit(int vCount, int eCount, MessageValueType *mValues) override;

    void IterationEnd(MessageValueType *mValues) override;

    void Deploy(int vCount, int eCount, int numOfInitV) override;

    void Free() override;

    int MSGApply_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                       MessageValueType *mValues) override;

    int MSGGenMerge_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                          MessageValueType *mValues) override;

protected:
    int vertexLimit;
    int mPerMSGSet;
    int ePerEdgeSet;

    ComputeUnit<double> *d_computeUnits;
    unsigned long long int *d_mTransformedMergedMSGValueSet;

private:

    auto MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                ComputeUnit<VertexValueType> *computeUnits);

    auto MSGApply_GPU_VVCopy(int computeUnitCount,
                             ComputeUnit<VertexValueType> *computeUnits);
};

#endif //GRAPH_ALGO_BELLMANFORDGPU_H
