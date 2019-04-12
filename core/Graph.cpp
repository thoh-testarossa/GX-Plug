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

Graph::Graph(int vCount)
{
    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();
    this->verticeValue = std::vector<double>();

    this->vCount = vCount;
    for(int i = 0; i < vCount; i++) this->vList.emplace_back(Vertex(i, false, INVALID_INITV_INDEX));
    this->eCount = 0;

    this->verticeValue_IPCArray_ptr = nullptr;
}

Graph::Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<double> &verticeValue)
{
    this->vCount = vSet.size();
    this->eCount = eSet.size();
    this->vList = vSet;
    this->eList = eSet;
    this->verticeValue = verticeValue;
    this->verticeValue_IPCArray_ptr = nullptr;
}

Graph::Graph(int vCount, int eCount, int numOfInitV, double *vValues, int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet)
{
    this->vCount = vCount;
    this->eCount = eCount;

    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();
    this->verticeValue = std::vector<double>();

    //v assemble
    //initVIndex will be initialized after other initializations finished
    for(int i = 0; i < this->vCount; i++)
    {
        auto v = Vertex(i, AVCheckSet[i], INVALID_INITV_INDEX);
        this->vList.emplace_back(v);
    }
    for(int i = 0; i < numOfInitV; i++)
        this->vList.at(initVSet[i]).initVIndex = i;

    //vValues assemble
    this->verticeValue.reserve(vCount * numOfInitV);
    this->verticeValue.assign(&vValues[0], &vValues[vCount * numOfInitV]);

    //e assemble
    for(int i = 0; i < this->eCount; i++)
        this->eList.emplace_back(eSrcSet[i], eDstSet[i], eWeightSet[i]);
}

void Graph::insertEdge(int src, int dst, double weight)
{
    this->eList.emplace_back(Edge(src, dst, weight));
    this->eCount++;
}
