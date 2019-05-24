//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPH_H
#define GRAPH_ALGO_GRAPH_H

#include "../include/deps.h"

#define INVALID_INITV_INDEX -1

class Vertex
{
public:
    Vertex(int vertexID, bool activeness, int initVIndex);

    int vertexID;
    bool isActive;
    int initVIndex;
};

class Edge
{
public:
    Edge(int src, int dst, double weight);

    int src;
    int dst;
    double weight;
};

class Graph
{
public:
    Graph(int vCount);
    Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<double> &verticeValue);
    Graph(int vCount, int eCount, int numOfInitV, double *vValues, int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet);

    void insertEdge(int src, int dst, double weight);

    int vCount;
    int eCount;

    std::vector<Vertex> vList;
    std::vector<Edge> eList;

    std::vector<double> verticeValue;
    std::vector<int> verticeLabelCnt;
    double *verticeValue_IPCArray_ptr;
};

#endif //GRAPH_ALGO_GRAPH_H
