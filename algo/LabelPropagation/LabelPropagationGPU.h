//
// Created by cave-g-f on 2019-10-22
//

#pragma once

#ifndef GRAPH_ALGO_LABELPROPAGATIONGPU_H
#define GRAPH_ALGO_LABELPROPAGATIONGPU_H

#include "LabelPropagation.h"
#include "../../include/GPUconfig.h"
#include "kernel_src/LabelPropagationGPU_kernel.h"

template <typename VertexValueType, typename MessageValueType>
class LabelPropagationGPU : public LabelPropagation<VertexValueType, MessageValueType>
{
public:
    LabelPropagationGPU();

    void Init(int vCount, int eCount, int numOfInitV, int computeUnitsCnt = 0) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
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

    ComputeUnit<VertexValueType> *d_computeUnits;
    MessageValueType *d_mTransformedMergedMSGValueSet;


private:

    auto MSGGenMerge_GPU_MVCopy(int computeUnitCount,
                                ComputeUnit<VertexValueType> *computeUnits);

    auto MSGApply_GPU_VVCopy(int computeUnitCount,
                             ComputeUnit<VertexValueType> *computeUnits);
};

#endif //GRAPH_ALGO_LabelPropagationGPU_H
