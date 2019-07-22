//
// Created by Thoh Testarossa on 2019-03-12
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_H
#define GRAPH_ALGO_BELLMANFORDGPU_H

#include "BellmanFord.h"

template <typename VertexValueType>
class BellmanFordGPU : public BellmanFord<VertexValueType>
{
public:
    BellmanFordGPU();

    void Init(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int numOfInitV) override;
    void Free() override;

    void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<VertexValueType> &mSet) override;
    void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<VertexValueType> &mSet) override;

    void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues) override;

protected:
    int vertexLimit;
    int mPerMSGSet;
    int ePerEdgeSet;

    int *initVSet;
    int *d_initVSet;
    int *initVIndexSet;
    int *d_initVIndexSet;
    double *vValueSet;
    double *d_vValueSet;

    VertexValueType *mValueTable;

    int *mInitVIndexSet;
    int *d_mInitVIndexSet;
    int *mDstSet;
    int *d_mDstSet;
    double *mValueSet;
    double *d_mValueSet;

    bool *AVCheckSet;
    bool *d_AVCheckSet;

    int *eSrcSet;
    int *d_eSrcSet;
    int *eDstSet;
    int *d_eDstSet;
    double *eWeightSet;
    double *d_eWeightSet;

    Vertex *d_vSet;
    Edge *d_eGSet;

    int *activeVerticesSet;
    int *d_activeVerticesSet;

    VertexValueType *mMergedMSGValueSet;
    unsigned long long int *mTransformedMergedMSGValueSet;
    unsigned long long int *d_mTransformedMergedMSGValueSet;

    unsigned long long int *mValueTSet;
    unsigned long long int *d_mValueTSet;

private:
    auto MSGGenMerge_GPU_MVCopy(Vertex *d_vSet, const Vertex *vSet,
                                double *d_vValues, const double *vValues,
                                unsigned long long int *d_mTransformedMergedMSGValueSet,
                                unsigned long long int *mTransformedMergedMSGValueSet,
                                int vGCount, int numOfInitV);

    auto MSGApply_GPU_VVCopy(Vertex *d_vSet, const Vertex *vSet,
                             double *d_vValues, const double *vValues,
                             int vGCount, int numOfInitV);
};

#endif //GRAPH_ALGO_BELLMANFORDGPU_H
