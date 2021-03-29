//
// Created by cave-g-f on 2019-9-21
//

#include "PageRank.h"

#include <iostream>
#include <chrono>
#include <algorithm>

template <typename VertexValueType, typename MessageValueType>
PageRank<VertexValueType, MessageValueType>::PageRank()
{
    this->resetProb = 0.15;
    this->deltaThreshold = 0.001;
}

template <typename VertexValueType, typename MessageValueType>
int PageRank<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if(g.eCount <= 0 || g.vCount <= 0) return 0;

    auto msgSize = mSet.mSet.size();

    //mValues init
    MessageValueType *mValues = new MessageValueType [msgSize];

    for(int i = 0; i < msgSize; i++)
    {
        mValues[i] = mSet.mSet.at(i).value;
    }

    //array form computation
//    auto avCount = this->MSGApply_array(g.vCount, msgSize, &g.vList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);

    //test
//    std::cout << "=============apply info================" << std::endl;
//    for(int i = 0; i < g.vCount; i++)
//    {
//        std::cout << g.vList[i].isActive << " " << g.verticesValue[i].first << " " << g.verticesValue[i].second << std::endl;
//    }
//    std::cout << "================end====================" << std::endl;

    //test
//    std::cout << avCount << std::endl;

    delete[] mValues;

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
int PageRank<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if(g.eCount <= 0 || g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType [g.vCount];

    int msgCnt = 0;
    //array form computation
//  msgCnt = this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);

    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(msgCnt);

    for(int i = 0; i < msgCnt; i++)
    {
        mSet.insertMsg(Message<MessageValueType>(0, mValues[i].destVId, mValues[i]));
    }

    //test
//    std::cout << "=============msg info============" << std::endl;
//    for(int i = 0; i < msgCnt; i++)
//    {
//        std::cout << mValues[i].destVId << " " << mValues[i].rank << std::endl;
//    }
//    std::cout << "===============end===============" << std::endl;

    delete[] mValues;

    return msgCnt;
}

template <typename VertexValueType, typename MessageValueType>
int PageRank<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
                                                                ComputeUnit<VertexValueType> *computeUnits,
                                                                MessageValueType *mValues)
{
    int avCount = 0;

    for (int i = 0; i < computeUnitCount; i++)
    {
        computeUnits[i].srcVertex.isActive = false;
        computeUnits[i].destVertex.isActive = false;
    }

    for (int i = 0; i < computeUnitCount; i++)
    {
        auto &computeUnit = computeUnits[i];
        int destVId = computeUnit.destVertex.vertexID;

        if (mValues[destVId].destVId == -1 || !computeUnit.destVertex.isMaster) continue;

        computeUnit.destVertex.isActive = true;
        computeUnit.destValue.first += mValues[destVId].rank;
        computeUnit.destValue.second = mValues[destVId].rank;
        avCount++;
    }

    std::cout << "avCount" << std::endl;

    return avCount;
}

template <typename VertexValueType, typename MessageValueType>
int PageRank<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                   ComputeUnit<VertexValueType> *computeUnits,
                                                                   MessageValueType *mValues)
{
    for (int i = 0; i < computeUnitCount; i++)
    {
        auto computeUnit = computeUnits[i];
        int destVId = computeUnit.destVertex.vertexID;

        if(computeUnit.srcValue.second > this->deltaThreshold)
        {
            double rank = computeUnit.srcValue.second * computeUnit.edgeWeight * (1 - this->resetProb);
            mValues[destVId].rank += rank;
            mValues[destVId].destVId = destVId;
        }
    }

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
std::vector<Graph<VertexValueType>> PageRank<VertexValueType, MessageValueType>::DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount)
{
    std::vector<Graph<VertexValueType>> res = std::vector<Graph<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) res.push_back(Graph<VertexValueType>(0));
    for(int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph<VertexValueType>(g.vList, std::vector<Edge>(), g.verticesValue);

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
        {
            res.at(i).eList.emplace_back(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
        }
        res.at(i).eCount = res.at(i).eList.size();
    }

    return res;
}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV, int maxComputeUnits)
{
    this->totalVValuesCount = vCount;
    this->totalMValuesCount = vCount;
    this->numOfInitV = numOfInitV;
    this->maxComputeUnits = maxComputeUnits;
}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    bool personalized = true;

    if(initVList.at(0) == INVALID_INITV_INDEX)
    {
        personalized = false;
    }

    if(personalized)
    {
        for(int i = 0; i < initVList.size(); i++)
        {
            g.vList.at(initVList.at(i)).initVIndex = initVList.at(i);
            g.vList.at(initVList.at(i)).isActive = true;
        }
    }
    else
    {
        for(int i = 0; i < g.vCount; i++)
        {
            g.vList.at(i).isActive = true;
        }
    }

    //vValues init
    g.verticesValue.reserve(g.vCount);
    g.verticesValue.assign(g.vCount, VertexValueType(0.0, 0.0));

    if(personalized)
    {
        for(int i = 0; i < g.vList.size(); i++)
        {
            if(g.vList.at(i).initVIndex == INVALID_INITV_INDEX)
            {
                g.verticesValue.at(i) = VertexValueType(0.0, 0.0);
            }
            else
            {
                g.verticesValue.at(i) = VertexValueType(1.0, 1.0);
            }
        }
    }
    else
    {
        for(int i = 0; i < g.vCount; i++)
        {
            auto newRank = (this->resetProb);
            g.verticesValue.at(i) = VertexValueType(newRank, newRank);
        }
    }

    //eValues init
    for(auto &e : g.eList)
    {
        e.weight = 1.0 / g.vList.at(e.src).outDegree;
    }
}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::Free()
{

}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                                                                     std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                                                                     const std::vector<int> &initVList)
{
    //init
    g.verticesValue.assign(g.vCount, VertexValueType(0.0, 0.0));

    for(int i = 0; i < g.vCount; i++)
    {
        g.vList.at(i).isActive = false;
    }

    //Merge graphs
    for(const auto &subG : subGSet)
    {
        for(int i = 0; i < subG.verticesValue.size(); i++)
        {
            if(subG.vList.at(i).isActive)
            {
                if(!g.vList.at(i).isActive)
                {
                    g.vList.at(i).isActive = subG.vList.at(i).isActive;
                    g.verticesValue.at(i).first = subG.verticesValue.at(i).first;
                    g.verticesValue.at(i).second = subG.verticesValue.at(i).second;
                }
                else
                {
                    g.verticesValue.at(i).first += subG.verticesValue.at(i).second;
                    g.verticesValue.at(i).second += subG.verticesValue.at(i).second;
                }
            }
            else if(!g.vList.at(i).isActive)
            {
                g.verticesValue.at(i) = subG.verticesValue.at(i);
            }
        }
    }

}

template<typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::IterationInit(int vCount, int eCount, MessageValueType *mValues)
{
    for(int i = 0; i < vCount; i++)
    {
        mValues[i] = MessageValueType(-1, 0);
    }
}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    MessageValueType *mValues = new MessageValueType[g.vCount * this->numOfInitV];

    int computeCnt = 0;

    std::vector<ComputeUnitPackage<VertexValueType>> computePackages;

    ComputeUnit<VertexValueType> *computeUnits = nullptr;

    for (int i = 0; i < g.eCount; i++)
    {
        int destVId = g.eList[i].dst;
        int srcVId = g.eList[i].src;

        if (!g.vList[srcVId].isActive) continue;

        if (computeCnt == 0) computeUnits = new ComputeUnit<VertexValueType>[this->maxComputeUnits];
        computeUnits[computeCnt].destVertex = g.vList[destVId];
        computeUnits[computeCnt].destValue = g.verticesValue[destVId];
        computeUnits[computeCnt].srcVertex = g.vList[srcVId];
        computeUnits[computeCnt].srcValue = g.verticesValue[srcVId];
        computeUnits[computeCnt].edgeWeight = g.eList[i].weight;
        computeCnt++;

        if (computeCnt == this->maxComputeUnits || i == g.eCount - 1)
        {
            computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
            computeCnt = 0;
        }
    }

    if (computeCnt != 0)
    {
        computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
    }

    for (auto &v : g.vList) v.isActive = false;
    activeVertices.clear();

    this->IterationInit(g.vCount, g.eCount, mValues);

    for (auto &computePackage : computePackages)
    {
        computeCnt = computePackage.getCount();
        computeUnits = computePackage.getUnitPtr();

        MSGGenMerge_array(computeCnt, computeUnits, mValues);
        MSGApply_array(computeCnt, computeUnits, mValues);

        for (int i = 0; i < computeCnt; i++)
        {
            int destVId = computeUnits[i].destVertex.vertexID;
            int srcVId = computeUnits[i].srcVertex.vertexID;
            int indexOfInit = computeUnits[i].indexOfInitV;

            g.vList[destVId].isActive |= computeUnits[i].destVertex.isActive;
            g.vList[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

            if (g.vList[destVId].isActive) activeVertices.emplace(destVId);
            if (g.vList[srcVId].isActive) activeVertices.emplace(srcVId);

            g.verticesValue[destVId] = computeUnits[i].destValue;
        }
    }

    std::cout << "avcount: " << activeVertices.size() << std::endl;

    free(mValues);
    for (auto &computePackage : computePackages)
    {
        free(computePackage.getUnitPtr());
    }
}

template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet<MessageValueType> mGenSet = MessageSet<MessageValueType>();
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
}


template <typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{

    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<MessageValueType>> mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    std::vector<MessageSet<MessageValueType>> mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size(), 100);

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    bool isActive = true;

    while(isActive)
    {
        isActive = false;

        std::cout << "iterCount: " << iterCount << std::endl;
        auto start = std::chrono::system_clock::now();
        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);
        auto divideGraphFinish = std::chrono::system_clock::now();

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertice.clear();

        auto mergeGraphStart = std::chrono::system_clock::now();
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        iterCount++;
        auto end = std::chrono::system_clock::now();

        for(int i = 0; i < g.vCount; i++)
        {
            if(g.vList.at(i).isActive)
            {
                isActive = true;
                break;
            }
        }

        //test
        for(int i = 0; i < g.vCount; i++)
        {
            std::cout << i << " " << g.verticesValue.at(i).first << " " << g.verticesValue.at(i).second << std::endl;
        }

        //time test
        std::cout << "merge time: " <<  std::chrono::duration_cast<std::chrono::microseconds>(end - mergeGraphStart).count() << std::endl;
    }

//    for(int i = 0; i < g.vCount; i++)
//    {
//        std::cout << i << " " << g.verticesValue.at(i).first << " " << g.verticesValue.at(i).second << std::endl;
//    }

    Free();
}

template<typename VertexValueType, typename MessageValueType>
void PageRank<VertexValueType, MessageValueType>::download(VertexValueType *vValues, Vertex *vSet, int computeUnitCount,
                                                           ComputeUnit<VertexValueType> *computeUnits)
{
    for (int i = 0; i < computeUnitCount; i++)
    {
        int destVId = computeUnits[i].destVertex.vertexID;
        int srcVId = computeUnits[i].srcVertex.vertexID;

        vSet[destVId].isActive |= computeUnits[i].destVertex.isActive;
        vSet[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

        vValues[destVId] = computeUnits[i].destValue;
    }
}

