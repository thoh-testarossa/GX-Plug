//
// Created by cave-g-f on 2019-05-17.
//

#pragma once

#ifndef GRAPH_ALGO_LABELPROPAGATION_H
#define GRAPH_ALGO_LABELPROPAGATION_H

#include "../../core/GraphUtil.h"

template <typename VertexValueType>
class LabelPropagation : public GraphUtil<VertexValueType>
{
public:
    LabelPropagation();

    void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<VertexValueType> &mSet) override;
    void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<VertexValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    void MSGInit_array(VertexValueType *mValues, int eCount, int vCount, int numOfInitV) override;
    void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues) override;

    void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;
    void MergeMergedMSG(MessageSet<VertexValueType> &mergedMSG, const std::vector<MessageSet<VertexValueType>> &mergedMSGSet) override;

    void Init(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);
    void Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList);

    void ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);
};

#endif
