//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/Graph.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../algo/LabelPropagation/LabelPropagation.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <future>
#include <cstring>

template<typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType> *uc, Graph<LPA_Value> &graph, Graph<LPA_Value> &subGraph,
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

    uc->transfer(&computePackages[0], computePackages.size());
    uc->startPipeline();

    int totalCnt = 0;
    for (auto &computePackage : computePackages)
    {
        computeCnt = computePackage.getCount();
        computeUnits = computePackage.getUnitPtr();

        for (int i = 0; i < computeCnt; i++, totalCnt++)
        {
            subGraph.verticesValue[totalCnt] = computeUnits[i].destValue;
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
    std::ifstream Gin(argv[1]);
    if (!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    // input the graph
    Graph<LPA_Value> test = Graph<LPA_Value>(vCount);
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

    //vValues init
    test.verticesValue.reserve(std::max(vCount, eCount));
    test.verticesValue.assign(std::max(vCount, eCount), LPA_Value(INVALID_INITV_INDEX, -1, 0));
    for (int i = 0; i < test.vCount; i++)
    {
        test.verticesValue.at(i) = LPA_Value(test.vList.at(i).vertexID, test.vList.at(i).vertexID, 0);
    }


    //partition
    std::vector<Graph<LPA_Value>> subGraph = std::vector<Graph<LPA_Value>>();
    for (int i = 0; i < nodeCount; i++) subGraph.push_back(Graph<LPA_Value>(0));
    for (int i = 0; i < nodeCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        subGraph.at(i) = Graph<LPA_Value>(test.vList, std::vector<Edge>(), test.verticesValue);

        //Distribute e info
        for (int k = i * test.eCount / nodeCount; k < (i + 1) * test.eCount / nodeCount; k++)
        {
            subGraph.at(i).eList.emplace_back(test.eList.at(k).src, test.eList.at(k).dst, test.eList.at(k).weight);
        }
        subGraph.at(i).eCount = subGraph.at(i).eList.size();
    }

    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient<LPA_Value, LPA_MSG>>();
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

    int iterCount = 0;

    //Test
    std::cout << "Init finished" << std::endl;
    //Test end

    while (iterCount < 40)
    {
        //Test
        std::cout << "Processing at iter " << ++iterCount << std::endl;
        //Test end

        auto labelCntPerVertice = std::vector<std::map<int, int>>();
        auto maxLabelCnt = std::vector<std::pair<int, int>>();

        labelCntPerVertice.reserve(vCount);
        labelCntPerVertice.assign(vCount, std::map<int, int>());
        maxLabelCnt.reserve(vCount);
        maxLabelCnt.assign(vCount, std::pair<int, int>(0, 0));

        auto futList = new std::future<void>[nodeCount];
        for (int i = 0; i < nodeCount; i++)
        {
            std::future<void> tmpFut = std::async(testFut<LPA_Value, LPA_MSG>, &clientVec.at(i), std::ref(test),
                                                  std::ref(subGraph[i]), (int)(1000));
            futList[i] = std::move(tmpFut);
        }

        for (int i = 0; i < nodeCount; i++)
            futList[i].get();

        std::cout << "merge the data" << std::endl;

        auto start = std::chrono::system_clock::now();

        for (const auto &subG : subGraph)
        {
            //vValues merge
            for (int i = 0; i < subG.eCount; i++)
            {
                auto lpaValue = subG.verticesValue.at(i);

                if (lpaValue.destVId == INVALID_INITV_INDEX)
                {
                    continue;
                }

                auto &labelCnt = labelCntPerVertice.at(lpaValue.destVId);
                auto &maxLabel = maxLabelCnt.at(lpaValue.destVId);

                if (labelCnt.find(lpaValue.label) == labelCnt.end())
                {
                    labelCnt[lpaValue.label] = 1;
                } else
                {
                    labelCnt[lpaValue.label] += 1;
                }

                if (maxLabel.second <= labelCnt[lpaValue.label])
                {
                    maxLabel.first = lpaValue.label;
                    maxLabel.second = labelCnt[lpaValue.label];
                }
            }
        }

        auto mergeEnd = std::chrono::system_clock::now();

        std::cout << "graph merge time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(mergeEnd - start).count() << std::endl;


        for (int i = 0; i < test.vCount; i++)
        {
            if (maxLabelCnt.at(i).second != 0)
            {
                test.verticesValue.at(i) = LPA_Value(i, maxLabelCnt.at(i).first, maxLabelCnt.at(i).second);
            } else
            {
                test.verticesValue.at(i) = LPA_Value(i, i, 0);
            }
        }
    }

    std::cout << "=========result=======" << std::endl;

    //result check
    for (int i = 0; i < vCount; i++)
    {
        std::cout << i << ":" << test.verticesValue[i].label << " " << test.verticesValue[i].labelCnt << std::endl;
    }

    for (int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();
}