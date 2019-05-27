//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "GraphUtil.h"

template <typename T>
std::vector<Graph<T>> GraphUtil<T>::DivideGraphByEdge(const Graph<T> &g, int partitionCount)
{
    std::vector<Graph<T>> res = std::vector<Graph<T>>();
    for(int i = 0; i < partitionCount; i++) res.push_back(Graph<T>(0));
    for(int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph<T>(g.vList, std::vector<Edge>(), g.verticeValue);

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
            res.at(i).insertEdge(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
    }

    return res;
}
