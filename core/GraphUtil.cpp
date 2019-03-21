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
        res.at(i) = Graph(g.vCount);
        //Copy v info
        for(int j = 0; j < g.vCount; j++)
        {
            if(res.at(i).vList.at(j).vertexID == g.vList.at(j).vertexID) //It should be of equal value
                res.at(i).vList.at(j).value = g.vList.at(j).value;
        }

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
            res.at(i).insertEdge(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
    }

    return res;
}
