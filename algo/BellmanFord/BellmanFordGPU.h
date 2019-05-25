//
// Created by Thoh Testarossa on 2019-03-12
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_H
#define GRAPH_ALGO_BELLMANFORDGPU_H

#include "BellmanFord.h"

template <typename T>
class BellmanFordGPU : public BellmanFord<T>
{
public:
    BellmanFordGPU();

    void Init(Graph<T> &g, std::set<int> &activeVertice, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int numOfInitV) override;
    void Free() override;

    void MSGApply(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<T> &mSet) override;
    void MSGGenMerge(const Graph<T> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<T> &mSet) override;

    void MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, T *vValues, T *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const T *vValues, T *mValues) override;


protected:
    int mPerMSGSet;
    int ePerEdgeSet;

    int *initVSet;
    int *d_initVSet;
    int *initVIndexSet;
    int *d_initVIndexSet;
    double *vValueSet;
    double *d_vValueSet;

    T *mValueTable;

    int *mInitVSet;
    int *d_mInitVSet;
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

    int *activeVerticeSet;
    int *d_activeVerticeSet;

    T *mMergedMSGValueSet;
    unsigned long long int *mTransformedMergedMSGValueSet;
    unsigned long long int *d_mTransformedMergedMSGValueSet;

    unsigned long long int *mValueTSet;
    unsigned long long int *d_mValueTSet;
};

#endif //GRAPH_ALGO_BELLMANFORDGPU_H
