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
    virtual void MSGApply(Graph &g, std::set<int> &activeVertices, const MessageSet &mSet) = 0;
    virtual void MSGGen(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet) = 0;
    virtual void MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source) = 0;
    virtual void MSGGenMerge(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet) = 0;

    //Master function
    virtual void Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList) = 0;
    virtual void Free(Graph &g) = 0;
    virtual void Deploy(Graph &g, int numOfInitV) = 0;
    virtual void MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
                            std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                            const std::vector<int> &initVList) = 0;
    virtual void MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet) = 0;

    std::vector<Graph> DivideGraphByEdge(const Graph &g, int partitionCount);
};

#endif //GRAPH_ALGO_GRAPHUTIL_H
