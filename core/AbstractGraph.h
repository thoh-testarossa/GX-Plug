//
// Created by Thoh Testarossa on 2019-07-19.
//

#ifndef GRAPH_ALGO_ABSTRACTGRAPH_H
#define GRAPH_ALGO_ABSTRACTGRAPH_H

#include "../include/deps.h"

#define INVALID_INITV_INDEX -1

class Vertex
{
public:
    Vertex(int vertexID, bool activeness, int initVIndex);

    int vertexID;
    bool isActive;
    bool needMerge;
    int initVIndex;
    int outDegree;
    int inDegree;
};

class Edge
{
public:
    Edge(int src, int dst, double weight);

    int src;
    int dst;
    double weight;
};

class AbstractGraph
{
public:
    AbstractGraph(int vCount);
    AbstractGraph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet);
    AbstractGraph(int vCount, int eCount, int *eSrcSet, int *eDstSet, double *eWeightSet);

    void insertEdge(int src, int dst, double weight);
    void insertEdgeWithVertexInfo(int src, int dst, double weight);

    int vCount;
    int eCount;

    std::vector<Vertex> vList;
    std::vector<Edge> eList;
};

#endif //GRAPH_ALGO_ABSTRACTGRAPH_H
