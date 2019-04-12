//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "GraphUtil.h"


std::vector<Graph> GraphUtil::DivideGraphByEdge(const Graph &g, int partitionCount)
{
    std::vector<Graph> res = std::vector<Graph>();
    for(int i = 0; i < partitionCount; i++) res.push_back(Graph(0));
    for(int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph(g.vList, std::vector<Edge>(), g.verticeValue);

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
            res.at(i).insertEdge(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
    }

    return res;
}
