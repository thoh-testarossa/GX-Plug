//
// Created by cave-g-f on 2019-9-23
//

#include "../algo/PageRank/PageRank.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<std::pair<double, double>> test = Graph<std::pair<double, double>>(vCount);
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

    PageRank<std::pair<double, double>, std::pair<int, double>> executor = PageRank<std::pair<double, double>, std::pair<int, double>>();
    //executor.Apply(test, initVList);
    executor.ApplyD(test, initVList, 4);
}

