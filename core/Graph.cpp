//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/Graph.h"

Vertex::Vertex(int vertexID, std::map<int, double> value)
{
    this->vertexID = vertexID;
    this->value = value;
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

    this->vCount = vCount;
    for(int i = 0; i < vCount; i++) this->vList.emplace_back(Vertex(i, std::map<int, double>()));
    this->eCount = 0;
}

Graph::Graph(int vCount, std::map<int, std::map<int, double>> vertex, std::vector<Edge> edge) {

    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();

    this->vCount = vCount;
    for(int i = 0; i < vCount; i++) this->vList.emplace_back(Vertex(i, std::map<int, double>()));
    this->eCount = 0;

    for(auto itE : edge){
        insertEdge(itE.src, itE.dst, itE.weight);
    }
    for(auto itV : vList){
        auto iter = vertex.find(itV.vertexID);
        if(iter!=vertex.end()){
            itV.value = vertex[itV.vertexID];
        }
    }
}

void Graph::insertEdge(int src, int dst, double weight)
{
    this->eList.emplace_back(Edge(src, dst, weight));
    this->eCount++;
}
