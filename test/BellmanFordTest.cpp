//
// Created by Thoh Testarossa on 2019-03-09.
//

#include "../algo/BellmanFord/BellmanFord.h"

#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
if(argc != 2)
    {
        std::cout << "Usage:" << std::endl << "./sssp_fpga graph_path vcount ecount" << std::endl;
        return 1;
    }

    //Read the Graph
    std::ifstream Gin(argv[1]);
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    // int vCount = atoi(argv[2]);
    // int eCount = atoi(argv[3]);
    int vCount,eCount;
    Gin >> vCount >> eCount;
    Graph<int> test = Graph<int>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        // Gin >> src >> dst >> weight;
        // test.insertEdge(src, dst, weight);

        Gin >> src >> dst;
        test.insertEdge(src, dst, 1);
        //for edge-cut partition
        test.vList.at(src).isMaster = true;
        test.vList.at(dst).isMaster = true;
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();
    initVList.push_back(1);

    auto executor = BellmanFord<int, int>();
    executor.ApplyD(test, initVList, 4);

    // for(int i = 0; i < test.vCount * initVList.size(); i++)
    // {
    //     if(i % initVList.size() == 0) std::cout << i / initVList.size() << ": ";
    //     std::cout << "(" << initVList.at(i % initVList.size()) << " -> " << test.verticesValue.at(i) << ")";
    //     if(i % initVList.size() == initVList.size() - 1) std::cout << std::endl;
    // }
}