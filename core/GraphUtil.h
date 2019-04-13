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
    virtual void MSGGenMerge(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet) = 0;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    virtual void MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, double *vValues, double *mValues) = 0;
    virtual void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const double *vValues, double *mValues) = 0;

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
