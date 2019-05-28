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

template <typename VertexValueType>
class Graph
{
public:
    Graph(int vCount);
    Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<VertexValueType> &verticesValue);
    Graph(int vCount, int eCount, int numOfInitV, int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet);

    void insertEdge(int src, int dst, double weight);

    int vCount;
    int eCount;

    std::vector<Vertex> vList;
    std::vector<Edge> eList;
    std::vector<VertexValueType> verticesValue;
};

#endif //GRAPH_ALGO_GRAPH_H
