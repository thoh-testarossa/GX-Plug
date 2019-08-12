//
// Created by Thoh Testarossa on 2019-08-12.
//

#pragma once

#ifndef GRAPH_ALGO_CONNECTEDCOMPONENTGPU_H
#define GRAPH_ALGO_CONNECTEDCOMPONENTGPU_H

#include "ConnectedComponent.h"
#include "../../include/GPUconfig.h"

template <typename VertexValueType>
class ConnectedComponentGPU : public ConnectedComponent<VertexValueType>
{
public:
    ConnectedComponentGPU();

    void Init(int vCount, int eCount, int numOfInitV) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<VertexValueType> &mSet) override;
    void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<VertexValueType> &mSet) override;

    void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues) override;

protected:
    int vertexLimit;
    int mPerMSGSet;
    int ePerEdgeSet;

    VertexValueType *vValueSet;
    double *d_vValueSet;

    VertexValueType *mValueTable;
    int *d_mValueTable;

    int *mInitVIndexSet;
    int *d_mInitVIndexSet;
    int *mDstSet;
    int *d_mDstSet;
    VertexValueType *mValueSet;
    double *d_mValueSet;

    Vertex *d_vSet;
    Edge *d_eGSet;
};

#endif //GRAPH_ALGO_CONNECTEDCOMPONENTGPU_H
