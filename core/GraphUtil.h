//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPHUTIL_H
#define GRAPH_ALGO_GRAPHUTIL_H

#include "../core/Graph.h"
#include "../core/MessageSet.h"

class GraphUtil
{
public:
    virtual void MSGApply(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet &mSet) = 0;
    //virtual void MSGGen(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet) = 0;
    //virtual void MSGMerge(const Graph &g, const std::vector<int> &initVSet, MessageSet &result, const MessageSet &source) = 0;
    virtual void MSGGenMerge(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet) = 0;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    virtual void MSGApply_array(int vCount, int numOfInitV, const int *initVSet, bool *AVCheckSet, double *vValues, double *mValues, int *initVIndexSet) = 0;
    //virtual void MSGGen_array(int vCount, int eCount, int numOfInitV, const int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, int &numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, bool *AVCheckSet) = 0;
    //virtual void MSGMerge_array(int vCount, int numOfInitV, const int *initVSet, int numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, double *mValues, int *initVIndexSet) = 0;
    virtual void MSGGenMerge_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, Edge *eSet, double *mValues, bool *AVCheckSet) = 0;

    //Master function
    virtual void Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList) = 0;
    virtual void Free() = 0;
    virtual void Deploy(int vCount, int numOfInitV) = 0;
    virtual void MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
                            std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                            const std::vector<int> &initVList) = 0;
    virtual void MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet) = 0;

    std::vector<Graph> DivideGraphByEdge(const Graph &g, int partitionCount);
};

#endif //GRAPH_ALGO_GRAPHUTIL_H
