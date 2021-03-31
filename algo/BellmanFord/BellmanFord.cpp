//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

template<typename VertexValueType, typename MessageValueType>
BellmanFord<VertexValueType, MessageValueType>::BellmanFord()
{
}

template<typename VertexValueType, typename MessageValueType>
int
BellmanFord<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                         std::set<int> &activeVertice,
                                                         const MessageSet<MessageValueType> &mSet)
{
    //Activity reset
    activeVertice.clear();

    //Availability check
    if (g.vCount <= 0) return 0;

    //MSG Init
    MessageValueType *mValues = new MessageValueType[g.vCount * this->numOfInitV];

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
        mValues[i] = (MessageValueType) INVALID_MASSAGE;
    for (int i = 0; i < mSet.mSet.size(); i++)
    {
        auto &mv = mValues[mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex];
        if (mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }

    //array form computation
//    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);

    //Active vertices set assembly
    for (int i = 0; i < g.vCount; i++)
    {
        if (g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    free(mValues);

    return activeVertice.size();
}

template<typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g,
                                                                const std::vector<int> &initVSet,
                                                                const std::set<int> &activeVertice,
                                                                MessageSet<MessageValueType> &mSet)
{
    //Generate merged msgs directly

    //Availability check
    if (g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType[g.vCount * this->numOfInitV];

    //array form computation
//    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0],
//                            &g.verticesValue[0], mValues);

    //Package mMergedMSGValueSet to result mSet
    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        if (mValues[i] != (MessageValueType) INVALID_MASSAGE)
        {
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<MessageValueType>(initV, dst, mValues[i]));
        }
    }

    free(mValues);

    return mSet.mSet.size();
}

template<typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
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
        int destVId = computeUnit.destVertex.vertexID;;

        if (!computeUnit.destVertex.isMaster) continue;

        if (computeUnit.destValue > (VertexValueType) mValues[destVId * numOfInitV + computeUnit.indexOfInitV])
        {
            computeUnit.destValue = (VertexValueType) mValues[destVId * numOfInitV + computeUnit.indexOfInitV];
            computeUnit.destVertex.isActive = true;
//            std::cout << "isActive: " << computeUnit.destVertex.vertexID << std::endl;
            avCount++;
        }
    }

//    std::cout << "address: " << &computeUnits[0] << std::endl;

    return avCount;
}

template<typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                      ComputeUnit<VertexValueType> *computeUnits,
                                                                      MessageValueType *mValues)
{
    for (int i = 0; i < computeUnitCount; i++)
    {
        auto computeUnit = computeUnits[i];
        int destVId = computeUnit.destVertex.vertexID;
        if (mValues[destVId * numOfInitV + computeUnit.indexOfInitV] >
            (MessageValueType) computeUnit.srcValue + computeUnit.edgeWeight)
        {
            mValues[destVId * numOfInitV + computeUnit.indexOfInitV] =
                    (MessageValueType) computeUnit.srcValue + computeUnit.edgeWeight;
        }
    }

    return computeUnitCount;
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFord<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV, int maxComputeUnits)
{
    this->numOfInitV = numOfInitV;

    //Memory parameter init
    this->totalVValuesCount = vCount * numOfInitV;
    this->totalMValuesCount = vCount * numOfInitV;
    this->maxComputeUnits = maxComputeUnits;

    this->optimize = false;
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                               const std::vector<int> &initVList)
{
    int numOfInitV_init = initVList.size();

    //v Init
    for (int i = 0; i < numOfInitV_init; i++)
        g.vList.at(initVList.at(i)).initVIndex = i;
    for (auto &v : g.vList)
    {
        if (v.initVIndex != INVALID_INITV_INDEX)
        {
            activeVertices.insert(v.vertexID);
            v.isActive = true;
        } else v.isActive = false;
    }

    //vValues init
    g.verticesValue.reserve(g.vCount * numOfInitV_init);
    g.verticesValue.assign(g.vCount * numOfInitV_init, (VertexValueType) (INT32_MAX >> 1));
    for (int initID : initVList)
        g.verticesValue.at(initID * numOfInitV_init + g.vList.at(initID).initVIndex) = (VertexValueType) 0;
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Free()
{

}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g,
                                                                const std::vector<Graph<VertexValueType>> &subGSet,
                                                                std::set<int> &activeVertices,
                                                                const std::vector<std::set<int>> &activeVerticeSet,
                                                                const std::vector<int> &initVList)
{
    //Init
    activeVertices.clear();
    for (auto &v : g.vList) v.isActive = false;

    //Merge graphs
    for (const auto &subG : subGSet)
    {
        //vSet merge
        for (int i = 0; i < subG.vCount; i++)
            g.vList.at(i).isActive |= subG.vList.at(i).isActive;

        //vValues merge
        for (int i = 0; i < subG.verticesValue.size(); i++)
        {
            if (g.verticesValue.at(i) > subG.verticesValue.at(i))
                g.verticesValue.at(i) = subG.verticesValue.at(i);
        }
    }

    //Merge active vertices set
    for (const auto &AVs : activeVerticeSet)
    {
        for (auto av : AVs)
            activeVertices.insert(av);
    }
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::IterationInit(int vCount, int eCount, MessageValueType *mValues)
{
    for (int i = 0; i < vCount * numOfInitV; i++)
    {
        mValues[i] = (MessageValueType) INVALID_MASSAGE;
    }
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFord<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                          std::set<int> &activeVertices)
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
        for (int j = 0; j < this->numOfInitV; j++)
        {

            if (computeCnt == 0) computeUnits = new ComputeUnit<VertexValueType>[this->maxComputeUnits];
            computeUnits[computeCnt].destVertex = g.vList[destVId];
            computeUnits[computeCnt].destValue = g.verticesValue[destVId * this->numOfInitV + j];
            computeUnits[computeCnt].srcVertex = g.vList[srcVId];
            computeUnits[computeCnt].srcValue = g.verticesValue[srcVId * this->numOfInitV + j];
            computeUnits[computeCnt].edgeWeight = g.eList[i].weight;
            computeUnits[computeCnt].indexOfInitV = j;
            computeCnt++;
        }

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

            if (g.verticesValue[destVId * numOfInitV + indexOfInit] > computeUnits[i].destValue)
                g.verticesValue[destVId * numOfInitV + indexOfInit] = computeUnits[i].destValue;
        }
    }

    std::cout << "avcount: " << activeVertices.size() << std::endl;

    free(mValues);
    for (auto &computePackage : computePackages)
    {
        free(computePackage.getUnitPtr());
    }
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFord<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList,
                                                       int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for (int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    auto mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    auto mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size(), 10000);

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while (activeVertices.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for (int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            AVSet.at(i) = activeVertices;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for (int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertices.clear();
        MergeGraph(g, subGraphSet, activeVertices, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end
    }

    Free();

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFord<VertexValueType, MessageValueType>::download(VertexValueType *vValues, Vertex *vSet, int computeUnitCount,
                                                         ComputeUnit<VertexValueType> *computeUnits)
{
    for (int i = 0; i < computeUnitCount; i++)
    {
        int destVId = computeUnits[i].destVertex.vertexID;
        int srcVId = computeUnits[i].srcVertex.vertexID;
        int indexOfInit = computeUnits[i].indexOfInitV;

        vSet[destVId].isActive |= computeUnits[i].destVertex.isActive;
        vSet[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

        if (vValues[destVId * this->numOfInitV + indexOfInit] > computeUnits[i].destValue)
        {
            vValues[destVId * this->numOfInitV + indexOfInit] = computeUnits[i].destValue;
        }
    }
}

template<typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::IterationEnd(MessageValueType *mValues)
{

}
