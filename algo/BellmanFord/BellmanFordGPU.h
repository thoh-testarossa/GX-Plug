//
// Created by Thoh Testarossa on 2019-03-12
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_H
#define GRAPH_ALGO_BELLMANFORDGPU_H

#include "BellmanFord.h"

class BellmanFordGPU : public BellmanFord
{
public:
    BellmanFordGPU();

    void Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList) override;
    void Deploy(Graph &g, int numOfInitV) override;
    void Free(Graph &g) override;

    void MSGApply(Graph &g, std::set<int> &activeVertice, const MessageSet &mSet) override;
    void MSGGen(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet) override;
    void MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source) override;
    void MSGGenMerge(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet) override;

    void MSGApply_array(int vCount, int numOfInitV, int *initVSet, bool *AVCheckSet, double *vValues, double *mValues) override;
    void MSGGen_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, int &numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, bool *AVCheckSet) override;
    void MSGMerge_array(int vCount, int numOfInitV, int *initVSet, int numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, double *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, double *mValues, bool *AVCheckSet) override;


protected:
    int mPerMSGSet;
    int ePerEdgeSet;

    int *initVSet;
    int *d_initVSet;
    double *vValueSet;
    double *d_vValueSet;

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

    int *activeVerticeSet;
    int *d_activeVerticeSet;

    double *mMergedMSGValueSet;
    unsigned long long int *mTransformedMergedMSGValueSet;
    unsigned long long int *d_mTransformedMergedMSGValueSet;

    unsigned long long int *mValueTSet;
    unsigned long long int *d_mValueTSet;
};

#endif //GRAPH_ALGO_BELLMANFORDGPU_H
