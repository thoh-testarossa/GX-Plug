//
// Created by Thoh Testarossa on 2019-08-22.
//

#pragma once

#ifndef GRAPH_ALGO_DDFS_H
#define GRAPH_ALGO_DDFS_H

#include "../../core/GraphUtil.h"

//Some state bit
#define STATE_IDLE false
#define STATE_DISCOVERED true

#define MARK_VISITED 1
#define MARK_PARENT 2
#define MARK_SON 4

#define MSG_SEND_TOKEN 32
#define MSG_SEND_VISITED 64
#define MSG_SEND_RESET 31

//DFS value class definition
//Assume this is corresponding to ith vertex
//As a message value, vStateList.second means which msgs vertex i will send to vStateList.first
//As a vertex value, vStateList.second means which marks vertex i marked for vStateList.first and the msg type vertex i will send to vStateList.first
class DFSValue
{
public:
    bool state;
    int parentIndex;
    int startTime;
    int endTime;
    int relatedVCount;
    std::vector<std::pair<int, char>> vStateList;
};

template <typename VertexValueType>
class DDFS : public GraphUtil<VertexValueType>
{
public:
    DDFS();

    void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<VertexValueType> &mSet) override;
    void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<VertexValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues) override;

    void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;

    void Init(int vCount, int eCount, int numOfInitV) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);
    void Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList);

    void ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;

    //The whole process will end immediately when this function return -1
    int search(int vid, int numOfInitV, const int *initVSet, Vertex *vSet, VertexValueType *vValues);
};

#endif //GRAPH_ALGO_DDFS_H
