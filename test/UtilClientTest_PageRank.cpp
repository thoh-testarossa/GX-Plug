//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/Graph.h"
#include "../srv/UtilClient.h"
#include "../algo/PageRank/PageRank.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <future>

int optimize = 0;

template<typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType> *uc, Graph<std::pair<double, double>> &graph,
             Graph<std::pair<double, double>> &subGraph,
             int maxComputeUnits)
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

        if (computeCnt == 0) computeUnits = new ComputeUnit<VertexValueType>[maxComputeUnits];
        computeUnits[computeCnt].destVertex = subGraph.vList[destVId];
        computeUnits[computeCnt].destValue = subGraph.verticesValue[destVId];
        computeUnits[computeCnt].srcVertex = subGraph.vList[srcVId];
        computeUnits[computeCnt].srcValue = subGraph.verticesValue[srcVId];
        computeUnits[computeCnt].edgeWeight = subGraph.eList[i].weight;
        computeCnt++;

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

    for (auto &v : subGraph.vList)
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

            subGraph.vList[destVId].isActive |= computeUnits[i].destVertex.isActive;
            subGraph.vList[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

            subGraph.verticesValue[destVId] = computeUnits[i].destValue;
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
        std::cout << "Usage:" << std::endl << "./UtilClientTest_PageRank graph vCount eCount numOfInitV nodecount"
                  << std::endl;
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
    std::ifstream Gin(argv[1]);
    if (!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    //init v index
    std::cout << "init initVSet ..." << std::endl;

    initVSet[0] = -1;

    std::cout << "init vSet ..." << std::endl;

    // input the graph
    Graph<std::pair<double, double>> test = Graph<std::pair<double, double>>(vCount);
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
    bool personalized = true;
    if (initVSet[0] == INVALID_INITV_INDEX)
    {
        personalized = false;
    }

    if (personalized)
    {
        for (int i = 0; i < numOfInitV; i++)
        {
            test.vList.at(initVSet[i]).initVIndex = initVSet[i];
            test.vList.at(initVSet[i]).isActive = true;
        }
    } else
    {
        for (int i = 0; i < test.vCount; i++)
        {
            test.vList.at(i).isActive = true;
        }
    }

    //vValues init
    test.verticesValue.reserve(test.vCount);
    test.verticesValue.assign(test.vCount, std::pair<double, double>(0.0, 0.0));

    if (personalized)
    {
        for (int i = 0; i < test.vList.size(); i++)
        {
            if (test.vList.at(i).initVIndex == INVALID_INITV_INDEX)
            {
                test.verticesValue.at(i) = std::pair<double, double>(0.0, 0.0);
            } else
            {
                test.verticesValue.at(i) = std::pair<double, double>(1.0, 1.0);
            }
        }
    } else
    {
        for (int i = 0; i < test.vCount; i++)
        {
            auto newRank = (0.15);
            test.verticesValue.at(i) = std::pair<double, double>(newRank, newRank);
        }
    }

    //eValue init
    for (auto &e : test.eList)
    {
        e.weight = 1.0 / test.vList.at(e.src).outDegree;
    }


    //partition
    std::vector<Graph<std::pair<double, double>>> subGraph = std::vector<Graph<std::pair<double, double>>>();
    for (int i = 0; i < nodeCount; i++) subGraph.emplace_back(0);
    for (int i = 0; i < nodeCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        subGraph.at(i) = Graph<std::pair<double, double>>(test.vList, std::vector<Edge>(), test.verticesValue);

        //Distribute e info
        for (int k = i * test.eCount / nodeCount; k < (i + 1) * test.eCount / nodeCount; k++)
            subGraph.at(i).eList.emplace_back(test.eList.at(k).src, test.eList.at(k).dst, test.eList.at(k).weight);

        subGraph.at(i).eCount = subGraph.at(i).eList.size();
    }


    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient<std::pair<double, double>, PRA_MSG>>();
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
            std::future<void> tmpFut = std::async(testFut<std::pair<double, double>, PRA_MSG>, &clientVec.at(i),
                                                  std::ref(test),
                                                  std::ref(subGraph[i]), (int) (100));
            futList[i] = std::move(tmpFut);
        }

        for (int i = 0; i < nodeCount; i++)
            futList[i].get();

        //Retrieve data
        isActive = false;
        for (auto &v : test.vList) v.isActive = false;

        for (const auto &subG : subGraph)
        {
            for (int i = 0; i < subG.verticesValue.size(); i++)
            {
                isActive |= subG.vList.at(i).isActive;
                if (subG.vList.at(i).isActive)
                {
                    if (!test.vList.at(i).isActive)
                    {
                        test.vList.at(i).isActive = subG.vList.at(i).isActive;
                        test.verticesValue.at(i).first = subG.verticesValue.at(i).first;
                        test.verticesValue.at(i).second = subG.verticesValue.at(i).second;
                    } else
                    {
                        test.verticesValue.at(i).first += subG.verticesValue.at(i).second;
                        test.verticesValue.at(i).second += subG.verticesValue.at(i).second;
                    }
                } else if (!test.vList.at(i).isActive)
                {
                    test.verticesValue.at(i) = subG.verticesValue.at(i);
                }
            }
        }

        int avCount = 0;
        for (auto v : test.vList)
        {
            if (v.isActive) avCount++;
        }

        std::cout << "avcount: " << avCount << std::endl;

    }

    std::cout << "===========result===========" << std::endl;
    //result check
    for (int i = 0; i < vCount; i++)
    {
        std::cout << i << ":" << test.verticesValue[i].first << " " << test.verticesValue[i].second << std::endl;
    }

    for (int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();
}