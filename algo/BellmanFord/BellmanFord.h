//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORD_H
#define GRAPH_ALGO_BELLMANFORD_H

#include "../../core/GraphUtil.h"

template <typename VertexValueType, typename MessageValueType>
class BellmanFord : public GraphUtil<VertexValueType, MessageValueType>
{
public:
    BellmanFord();

    int MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet) override;
    int MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    int MSGApply_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits, MessageValueType *mValues) override;
    int MSGGenMerge_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits, MessageValueType *mValues) override;

    void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;

    void Init(int vCount, int eCount, int numOfInitV, int maxComputeUnits=0) override;
    void IterationInit(int vCount, int eCount, MessageValueType *mValues) override;
    void IterationEnd(MessageValueType *mValues) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    void download(VertexValueType *vValues, Vertex *vSet, int computeUnitCount,
                  ComputeUnit<VertexValueType> *computeUnits) override;

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);

    void ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;
};

#endif //GRAPH_ALGO_BELLMANFORD_H
