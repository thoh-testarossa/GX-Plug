//
// Created by Thoh Testarossa on 2019-03-12.
//

#include "../algo/BellmanFord/BellmanFordGPU.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph test = Graph(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();
    initVList.push_back(1);
    initVList.push_back(2);
    initVList.push_back(4);

    BellmanFordGPU executor = BellmanFordGPU();
    //executor.Apply(test, initVList);
    executor.ApplyD(test, initVList, 4);

    for(auto v : test.vList)
    {
        std::cout << v.vertexID << ": ";
        for(auto iter = v.value.begin(); iter != v.value.end(); iter++)
            std::cout << "(" << iter->first << " -> " << iter->second << ")";
        std::cout << std::endl;
    }
    return 0;
}