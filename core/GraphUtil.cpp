//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "GraphUtil.h"

#define NO_REFLECTION -1

template <typename VertexValueType>
std::vector<Graph<VertexValueType>> GraphUtil<VertexValueType>::DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount)
{
    std::vector<Graph<VertexValueType>> res = std::vector<Graph<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) res.push_back(Graph<VertexValueType>(0));
    for(int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph<VertexValueType>(g.vList, std::vector<Edge>(), g.verticesValue);

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
            res.at(i).insertEdge(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
    }

    return res;
}

template<typename VertexValueType>
Graph<VertexValueType>
GraphUtil<VertexValueType>::reflect(const Graph<VertexValueType> &o_g, const std::vector<Edge> &eSet, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex)
{
    //Init
    int vCount = o_g.vCount;
    int eCount = eSet.size();

    reflectIndex.clear();
    reversedIndex.clear();
    reflectIndex.reserve(2 * eCount * sizeof(int));
    reversedIndex.reserve(vCount * sizeof(int));

    reversedIndex.assign(vCount, NO_REFLECTION);

    //Calculate reflection using eSet and generate reflected eSet
    auto r_eSet = std::vector<Edge>();
    r_eSet.reserve(2 * eCount * sizeof(Edge));

    int reflectCount = 0;

    for(const auto &e : eSet)
    {
        if(reversedIndex.at(e.src) == NO_REFLECTION)
        {
            reflectIndex.emplace_back(e.src);
            reversedIndex.at(e.src) = reflectCount;

            reflectCount++;
        }
        if(reversedIndex.at(e.dst) == NO_REFLECTION)
        {
            reflectIndex.emplace_back(e.dst);
            reversedIndex.at(e.dst) = reflectCount;

            reflectCount++;
        }

        r_eSet.emplace_back(reversedIndex.at(e.src), reversedIndex.at(e.dst), e.weight);
    }

    //Generate reflected vSet & vValueSet
    auto r_vSet = std::vector<Vertex>();
    r_vSet.reserve(reflectCount * sizeof(Vertex));

    int numOfInitV = o_g.verticesValue.size() / o_g.vCount;
    auto r_vValueSet = std::vector<VertexValueType>();
    r_vValueSet.reserve(reflectCount * numOfInitV * sizeof(VertexValueType));

    for(int i = 0; i < reflectCount; i++)
    {
        r_vSet.emplace_back(o_g.vList.at(reflectIndex.at(i)));
        for(int j = 0; j < numOfInitV; j++)
            r_vValueSet.emplace_back(o_g.verticesValue.at(reflectIndex.at(i) * numOfInitV + j));

        r_vSet.at(i).vertexID = i;
    }

    //Generate reflected graph and return
    return Graph<VertexValueType>(r_vSet, r_eSet, r_vValueSet);
}
