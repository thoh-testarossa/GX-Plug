//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPH_H
#define GRAPH_ALGO_GRAPH_H

#include "../include/deps.h"

class Vertex
{
public:
    Vertex(int vertexID, bool activeness, std::map<int, double> value);

    int vertexID;
    bool isActive;
    std::map<int, double> value;
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
    Graph(int vCount, std::map<int, std::map<int, double>> &vertex, std::vector<Edge> &edge, std::set<int>& activeID, std::set<int> &markID);

    void insertEdge(int src, int dst, double weight);

    int vCount;
    int eCount;

    std::vector<Vertex> vList;
    std::vector<Edge> eList;
};

#endif //GRAPH_ALGO_GRAPH_H
