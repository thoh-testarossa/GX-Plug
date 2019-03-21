//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORD_H
#define GRAPH_ALGO_BELLMANFORD_H

#include "../../core/GraphUtil.h"

class BellmanFord : public GraphUtil
{
public:
    BellmanFord();

    void MSGApply(Graph &g, std::set<int> &activeVertice, const MessageSet &mSet) override;
    void MSGGen(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet) override;
    void MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source) override;

    void MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
                    std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;
    void MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet) override;

    void Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList) override;
    void Deploy(Graph &g) override;
    void Free(Graph &g) override;

    void ApplyStep(Graph &g, std::set<int> &activeVertice);
    void Apply(Graph &g, const std::vector<int> &initVList);

    void ApplyD(Graph &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;
};

#endif //GRAPH_ALGO_BELLMANFORD_H
