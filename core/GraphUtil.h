//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPHUTIL_H
#define GRAPH_ALGO_GRAPHUTIL_H

#include "../core/Graph.h"
#include "../core/MessageSet.h"

template <typename VertexValueType>
class GraphUtil
{
public:
    virtual void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<VertexValueType> &mSet) = 0;
    virtual void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<VertexValueType> &mSet) = 0;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    virtual void MSGInit_array(VertexValueType *mValues, int eCount, int vCount, int numOfInitV) = 0;
    virtual void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues) = 0;
    virtual void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues) = 0;

    //Master function
    virtual void Init(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) = 0;
    virtual void Free() = 0;
    virtual void Deploy(int vCount, int numOfInitV) = 0;
    virtual void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                            std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                            const std::vector<int> &initVList) = 0;
    virtual void MergeMergedMSG(MessageSet<VertexValueType> &mergedMSG, const std::vector<MessageSet<VertexValueType>> &mergedMSGSet) = 0;

    std::vector<Graph<VertexValueType>> DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount);
};

#endif //GRAPH_ALGO_GRAPHUTIL_H
