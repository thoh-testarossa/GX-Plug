//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPHUTIL_H
#define GRAPH_ALGO_GRAPHUTIL_H

#include "../core/Graph.h"
#include "../core/MessageSet.h"

template <typename T>
class GraphUtil
{
public:
    virtual void MSGApply(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<T> &mSet) = 0;
    virtual void MSGGenMerge(const Graph<T> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<T> &mSet) = 0;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    virtual void MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, T *vValues, T *mValues) = 0;
    virtual void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const T *vValues, T *mValues) = 0;

    //Master function
    virtual void Init(Graph<T> &g, std::set<int> &activeVertice, const std::vector<int> &initVList) = 0;
    virtual void Free() = 0;
    virtual void Deploy(int vCount, int numOfInitV) = 0;
    virtual void MergeGraph(Graph<T> &g, const std::vector<Graph<T>> &subGSet,
                            std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                            const std::vector<int> &initVList) = 0;
    virtual void MergeMergedMSG(MessageSet<T> &mergedMSG, const std::vector<MessageSet<T>> &mergedMSGSet) = 0;

    std::vector<Graph<T>> DivideGraphByEdge(const Graph<T> &g, int partitionCount);
};

#endif //GRAPH_ALGO_GRAPHUTIL_H
