//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/Graph.h"

Vertex::Vertex(int vertexID, bool activeness, int initVIndex)
{
    this->vertexID = vertexID;
    this->isActive = activeness;
    this->initVIndex = initVIndex;
}

Edge::Edge(int src, int dst, double weight)
{
    this->src = src;
    this->dst = dst;
    this->weight = weight;
}

template <typename T>
Graph<T>::Graph(int vCount)
{
    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();
    this->verticeValue = std::vector<T>();

    this->vCount = vCount;
    for(int i = 0; i < vCount; i++) this->vList.emplace_back(Vertex(i, false, INVALID_INITV_INDEX));
    this->eCount = 0;
}

template <typename T>
Graph<T>::Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<T> &verticeValue)
{
    this->vCount = vSet.size();
    this->eCount = eSet.size();
    this->vList = vSet;
    this->eList = eSet;
    this->verticeValue = verticeValue;
}

template <typename T>
Graph<T>::Graph(int vCount, int eCount, int numOfInitV, int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet)
{
    this->vCount = vCount;
    this->eCount = eCount;

    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();
    this->verticeValue = std::vector<T>();

    //v assemble
    //initVIndex will be initialized after other initializations finished
    for(int i = 0; i < this->vCount; i++)
    {
        auto v = Vertex(i, AVCheckSet[i], INVALID_INITV_INDEX);
        this->vList.emplace_back(v);
    }
    for(int i = 0; i < numOfInitV; i++)
        this->vList.at(initVSet[i]).initVIndex = i;

    //e assemble
    for(int i = 0; i < this->eCount; i++)
        this->eList.emplace_back(eSrcSet[i], eDstSet[i], eWeightSet[i]);
}

template <typename T>
void Graph<T>::insertEdge(int src, int dst, double weight)
{
    this->eList.emplace_back(Edge(src, dst, weight));
    this->eCount++;
}
