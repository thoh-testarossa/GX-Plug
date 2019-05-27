//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORD_H
#define GRAPH_ALGO_BELLMANFORD_H

#include "../../core/GraphUtil.h"

template <typename T>
class BellmanFord : public GraphUtil<T>
{
public:
    BellmanFord();

    void MSGApply(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<T> &mSet) override;
    void MSGGenMerge(const Graph<T> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<T> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    void MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, T *vValues, T *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const T *vValues, T *mValues) override;

    void MergeGraph(Graph<T> &g, const std::vector<Graph<T>> &subGSet,
                    std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;
    void MergeMergedMSG(MessageSet<T> &mergedMSG, const std::vector<MessageSet<T>> &mergedMSGSet) override;

    void Init(Graph<T> &g, std::set<int> &activeVertice, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice);
    void Apply(Graph<T> &g, const std::vector<int> &initVList);

    void ApplyD(Graph<T> &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;
};

#endif //GRAPH_ALGO_BELLMANFORD_H
