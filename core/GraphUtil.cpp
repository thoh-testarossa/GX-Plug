//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "GraphUtil.h"

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
