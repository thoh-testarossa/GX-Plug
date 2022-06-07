#include "../algo/BellmanFord/BellmanFordFPGA.h"

#include <iostream>
#include <fstream>
#include <algorithm>
int main(int argc, char *argv[])
{
    // if(argc != 4)
    // {
    //     std::cout << "Usage:" << std::endl << "./sssp_fpga graph_path vcount ecount" << std::endl;
    //     return 1;
    // }

    // //Read the Graph
    // std::ifstream Gin(argv[1]);
    // if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    // int vCount = atoi(argv[2]);
    // int eCount = atoi(argv[3]);

    // Graph<int> test = Graph<int>(vCount);
    // for(int i = 0; i < eCount; i++)
    // {
    //     int src, dst;
    //     double weight;

    //     Gin >> src >> dst >> weight;
    //     test.insertEdge(src, dst, weight);

    //     //for edge-cut partition
    //     test.vList.at(src).isMaster = true;
    //     test.vList.at(dst).isMaster = true;
    // }

    // Gin.close();

    auto executor = BellmanFordFPGA<int, int>();
    
}