//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/Graph.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <future>
#include <cstring>

template<typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType> *uc, Graph<double> &graph, Graph<double> &subGraph, int maxComputeUnits)
{
    subGraph.vList = graph.vList;
    subGraph.verticesValue = graph.verticesValue;

    int computeCnt = 0;
    std::vector<ComputeUnitPackage<VertexValueType>> computePackages;
    ComputeUnit<VertexValueType> *computeUnits = nullptr;

    for (int i = 0; i < subGraph.eCount; i++)
    {
        int destVId = subGraph.eList[i].dst;
        int srcVId = subGraph.eList[i].src;

        if (!subGraph.vList[srcVId].isActive) continue;
        for (int j = 0; j < uc->numOfInitV; j++)
        {
            if (computeCnt == 0) computeUnits = new ComputeUnit<VertexValueType>[maxComputeUnits];
            computeUnits[computeCnt].destVertex = subGraph.vList[destVId];
            computeUnits[computeCnt].destValue = subGraph.verticesValue[destVId * uc->numOfInitV + j];
            computeUnits[computeCnt].srcVertex = subGraph.vList[srcVId];
            computeUnits[computeCnt].srcValue = subGraph.verticesValue[srcVId * uc->numOfInitV + j];
            computeUnits[computeCnt].edgeWeight = subGraph.eList[i].weight;
            computeUnits[computeCnt].indexOfInitV = j;
            computeCnt++;
        }

        if (computeCnt == maxComputeUnits || i == subGraph.eCount - 1)
        {
            computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
            computeCnt = 0;
        }
    }

    if (computeCnt != 0)
    {
        computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
    }

    for(auto &v : subGraph.vList)
        v.isActive = false;

    uc->transfer(&computePackages[0], computePackages.size());
    uc->startPipeline();

    for (auto &computePackage : computePackages)
    {
        computeCnt = computePackage.getCount();
        computeUnits = computePackage.getUnitPtr();

        for (int i = 0; i < computeCnt; i++)
        {
            int destVId = computeUnits[i].destVertex.vertexID;
            int srcVId = computeUnits[i].srcVertex.vertexID;
            int indexOfInit = computeUnits[i].indexOfInitV;

            subGraph.vList[destVId].isActive |= computeUnits[i].destVertex.isActive;
            subGraph.vList[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

            if (subGraph.verticesValue[srcVId * uc->numOfInitV + indexOfInit] > computeUnits[i].srcValue)
                subGraph.verticesValue[srcVId * uc->numOfInitV + indexOfInit] = computeUnits[i].srcValue;

            if (subGraph.verticesValue[destVId * uc->numOfInitV + indexOfInit] > computeUnits[i].destValue)
                subGraph.verticesValue[destVId * uc->numOfInitV + indexOfInit] = computeUnits[i].destValue;
        }
    }


    for (auto &computePackage : computePackages)
    {
        free(computePackage.getUnitPtr());
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4 && argc != 5 && argc != 6)
    {
        std::cout << "Usage:" << std::endl
                  << "./UtilClientTest_LabelPropagation graph vCount eCount numOfInitV nodecount" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[2]);
    int eCount = atoi(argv[3]);
    int numOfInitV = atoi(argv[4]);
    int nodeCount = atoi(argv[5]);

    //Parameter check
    if (vCount <= 0 || eCount <= 0 || numOfInitV <= 0 || nodeCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    //Init the Graph
    int *initVSet = new int[numOfInitV];
    bool *filteredV = new bool[vCount];
    int *timestamp = new int[vCount];

    std::ifstream Gin(argv[1]);
    if (!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    //init v index
    std::cout << "init initVSet ..." << std::endl;

    initVSet[0] = 1;

    std::cout << "init vSet ..." << std::endl;

    // input the graph
    Graph<double> test = Graph<double>(vCount);
    for (int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);

        test.vList.at(src).isMaster = true;
        test.vList.at(dst).isMaster = true;
    }

    Gin.close();

    //v Init
    for (int i = 0; i < numOfInitV; i++)
        test.vList.at(initVSet[i]).initVIndex = i;
    for (auto &v : test.vList)
    {
        if (v.initVIndex != INVALID_INITV_INDEX)
        {
            v.isActive = true;
        } else v.isActive = false;
    }

    //vValues init
    test.verticesValue.reserve(test.vCount * numOfInitV);
    test.verticesValue.assign(test.vCount * numOfInitV, (double) (INT32_MAX >> 1));
    for (int i = 0; i < numOfInitV; i++)
        test.verticesValue.at(initVSet[i] * numOfInitV + test.vList.at(initVSet[i]).initVIndex) = (double) 0;


    //partition
    std::vector<Graph<double>> subGraph = std::vector<Graph<double>>();
    for (int i = 0; i < nodeCount; i++) subGraph.emplace_back(0);
    for (int i = 0; i < nodeCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        subGraph.at(i) = Graph<double>(test.vList, std::vector<Edge>(), test.verticesValue);

        //Distribute e info
        for (int k = i * test.eCount / nodeCount; k < (i + 1) * test.eCount / nodeCount; k++)
            subGraph.at(i).eList.emplace_back(test.eList.at(k).src, test.eList.at(k).dst, test.eList.at(k).weight);

        subGraph.at(i).eCount = subGraph.at(i).eList.size();
        std::cout << subGraph.at(i).eCount << std::endl;
    }



    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient<double, double>>();
    for (int i = 0; i < nodeCount; i++)
        clientVec.emplace_back(numOfInitV, i, 2);
    int chk = 0;
    for (int i = 0; i < nodeCount; i++)
    {
        chk = clientVec.at(i).connect();
        if (chk == -1)
        {
            std::cout << "Cannot establish the connection with server correctly" << std::endl;
            return 2;
        }
    }

    bool isActive = true;
    int iterCount = 0;

    //Test
    std::cout << "Init finished" << std::endl;
    //Test end

    while (isActive)
    {
        //Test
        std::cout << "Processing at iter " << ++iterCount << std::endl;
        //Test end

        auto futList = new std::future<void>[nodeCount];
        for (int i = 0; i < nodeCount; i++)
        {
            std::future<void> tmpFut = std::async(testFut<double, double>, &clientVec.at(i), std::ref(test),
                                                  std::ref(subGraph[i]), (int)(1000));
            futList[i] = std::move(tmpFut);
        }

        for (int i = 0; i < nodeCount; i++)
            futList[i].get();


        //Retrieve data
        isActive = false;
        for (auto &v : test.vList) v.isActive = false;

        for (int i = 0; i < nodeCount; i++)
        {
            for (const auto &subG : subGraph)
            {
                //vSet merge
                for (int j = 0; j < subG.vCount; j++)
                {
                    test.vList.at(j).isActive |= subG.vList.at(j).isActive;
                    isActive |= subG.vList.at(j).isActive;
                }

                //vValues merge
                for (int j = 0; j < subG.verticesValue.size(); j++)
                {
                    if (test.verticesValue.at(j) > subG.verticesValue.at(j))
                        test.verticesValue.at(j) = subG.verticesValue.at(j);
                }
            }
        }

        int avCount = 0;
        for(auto v : test.vList)
        {
            if(v.isActive) avCount++;
        }

        std::cout << "avcount: " << avCount << std::endl;

    }

    for (int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();

    //result check
    for (int i = 0; i < vCount * numOfInitV; i++)
    {
        if (i % numOfInitV == 0) std::cout << i / numOfInitV << ": ";
        std::cout << "(" << initVSet[i % numOfInitV] << " -> " << test.verticesValue[i] << ")";
        if (i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
    }
}