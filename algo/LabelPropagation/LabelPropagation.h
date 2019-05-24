//
// Created by cave-g-f on 2019-05-17.
//

#pragma once

#ifndef GRAPH_ALGO_LABELPROPAGATION_H
#define GRAPH_ALGO_LABELPROPAGATION_H

#include "../../core/GraphUtil.h"

class LabelPropagation : public GraphUtil
{
public:
    LabelPropagation();

    void MSGApply(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet &mSet) override;
    void MSGGenMerge(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet) override;

    void MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, double *vValues, double *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const double *vValues, double *mValues) override;

    void MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
                    std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;
    void MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet) override;

    void Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice);
    void Apply(Graph &g, const std::vector<int> &initVList);

    void ApplyD(Graph &g, const std::vector<int> &initVList, int partitionCount);
};

#endif
