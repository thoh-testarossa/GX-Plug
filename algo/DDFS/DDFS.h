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

#define OP_BROADCAST 1
#define OP_MSG_FROM_SEARCH 2
#define OP_MSG_DOWNWARD 4

#define MARK_UNVISITED 0
#define MARK_VISITED 1
#define MARK_PARENT 2
#define MARK_SON 3

#define MSG_TOKEN 8
#define MSG_VISITED 16

#define MSG_SEND_TOKEN 32
#define MSG_SEND_VISITED 64
#define MSG_SEND_RESET 31

//DFS value class definition
class DFSValue
{
public:
    bool state;
    char opbit;
    int vNextMSGTo;
    int startTime;
    int endTime;
    int relatedVCount;
    std::vector<std::pair<int, char>> vStateList;
};

//DFS msg class definition
class DFSMSG
{
public:
    DFSMSG(int src, int dst, int timestamp, char msgbit)
    {
        this->src = src;
        this->dst = dst;
        this->timestamp = timestamp;
        this->msgbit = msgbit;
    }

    int src;
    int dst;
    int timestamp;
    char msgbit;
};

template <typename VertexValueType, typename MessageValueType>
class DDFS : public GraphUtil<VertexValueType, MessageValueType>
{
public:
    DDFS();

    void MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet) override;
    void MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    void MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues) override;
    void MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) override;

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
