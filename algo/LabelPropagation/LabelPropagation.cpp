//
// Created by cave-g-f on 2019-05-17.
//

#include "LabelPropagation.h"

#include <iostream>
#include <chrono>
#include <algorithm>

template<typename VertexValueType, typename MessageValueType>
LabelPropagation<VertexValueType, MessageValueType>::LabelPropagation()
{
}

template<typename VertexValueType, typename MessageValueType>
std::vector<Graph<VertexValueType>>
LabelPropagation<VertexValueType, MessageValueType>::DivideGraphByEdge(const Graph<VertexValueType> &g,
                                                                       int partitionCount)
{
    std::vector<Graph<VertexValueType>> res = std::vector<Graph<VertexValueType>>();
    for (int i = 0; i < partitionCount; i++) res.push_back(Graph<VertexValueType>(0));
    for (int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph<VertexValueType>(g.vList, std::vector<Edge>(), g.verticesValue);

        //Distribute e info
        for (int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
        {
            res.at(i).eList.emplace_back(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
        }
        res.at(i).eCount = res.at(i).eList.size();
    }
    return res;
}

template<typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g,
                                                                  const std::vector<int> &initVSet,
                                                                  std::set<int> &activeVertices,
                                                                  const MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if (g.eCount <= 0 || g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType[std::max(g.vCount, g.eCount)];

    for (int i = 0; i < std::max(g.vCount, g.eCount); i++)
    {
        mValues[i] = mSet.mSet.at(i).value;
    }

    //array form computation
    std::cout << "msg apply array..." << std::endl;
//    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);
    std::cout << "msg apply array end" << std::endl;

    delete[] mValues;

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g,
                                                                     const std::vector<int> &initVSet,
                                                                     const std::set<int> &activeVertice,
                                                                     MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if (g.eCount <= 0 || g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType[std::max(g.vCount, g.eCount)];

    //array form computation
    std::cout << "msg merge array..." << std::endl;
//    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, &initVSet[0], &g.verticesValue[0],
//                            mValues);
    std::cout << "msg merge array... end" << std::endl;

    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(std::max(g.vCount, g.eCount));

    for (int i = 0; i < std::max(g.vCount, g.eCount); i++)
    {
        mSet.insertMsg(Message<MessageValueType>(0, 0, mValues[i]));
    }

    delete[] mValues;

    return std::max(g.vCount, g.eCount);
}

template<typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply_array(int computeUnitCount,
                                                                        ComputeUnit<VertexValueType> *computeUnits,
                                                                        MessageValueType *mValues)
{
    for (int i = 0; i < computeUnitCount; i++)
    {
        auto &computeUnit = computeUnits[i];
        if (!computeUnit.destVertex.isMaster) continue;

        computeUnit.destValue.label = computeUnit.srcValue.label;
        computeUnit.destValue.destVId = computeUnit.destVertex.vertexID;
        computeUnit.destValue.labelCnt = 1;

        computeUnit.destVertex.isActive = true;
    }

    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge_array(int computeUnitCount,
                                                                           ComputeUnit<VertexValueType> *computeUnits,
                                                                           MessageValueType *mValues)
{
//    for(int i = eCount; i < vCount; i++)
//        mValues[i] = MessageValueType(-1, 0);
//
//    for(int i = 0; i < eCount; i++)
//    {
//        mValues[i] = MessageValueType(eSet[i].dst, vValues[eSet[i].src].label);
//    }
//
//    return std::max(vCount, eCount);

    for(int i = 0; i < computeUnitCount; i++)
    {
        int destVId = computeUnits[i].destVertex.vertexID;
        int srcVId = computeUnits[i].srcVertex.vertexID;

        mValues[this->pipeMsgCnt++] = MessageValueType(destVId, computeUnits[i].srcValue.label);
    }


    return computeUnitCount;
}

template<typename VertexValueType, typename MessageValueType>
void
LabelPropagation<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV, int maxComputeUnits)
{
    //vValue size = e in order to store the label info for merging the subgraph
    int max = vCount > eCount ? vCount : eCount;

    this->totalVValuesCount = max;
    this->totalMValuesCount = max;

    this->maxComputeUnits = maxComputeUnits;
}

template<typename VertexValueType, typename MessageValueType>
void
LabelPropagation<VertexValueType, MessageValueType>::IterationInit(int vCount, int eCount, MessageValueType *mValues)
{
    this->pipeDownloadCnt = 0;
    this->pipeMsgCnt = 0;
}

template<typename VertexValueType, typename MessageValueType>
void
LabelPropagation<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                               const std::vector<int> &initVList)
{
    //vValues init
    g.verticesValue.reserve(this->totalMValuesCount);
    g.verticesValue.assign(this->totalMValuesCount, LPA_Value(INVALID_INITV_INDEX, -1, 0));

    for (int i = 0; i < g.vCount; i++)
    {
        g.verticesValue.at(i) = LPA_Value(g.vList.at(i).vertexID, g.vList.at(i).vertexID, 0);
    }
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Free()
{
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g,
                                                                     const std::vector<Graph<VertexValueType>> &subGSet,
                                                                     std::set<int> &activeVertices,
                                                                     const std::vector<std::set<int>> &activeVerticeSet,
                                                                     const std::vector<int> &initVList)
{
    //init
    g.verticesValue.clear();
    g.verticesValue.assign(g.eCount, LPA_Value(INVALID_INITV_INDEX, -1, 0));

    //Merge graphs
    auto labelCntPerVertice = std::vector<std::map<int, int>>();
    auto maxLabelCnt = std::vector<std::pair<int, int>>();

    labelCntPerVertice.reserve(g.vCount);
    labelCntPerVertice.assign(g.vCount, std::map<int, int>());
    maxLabelCnt.reserve(g.vCount);
    maxLabelCnt.assign(g.vCount, std::pair<int, int>(0, 0));

    for (const auto &subG : subGSet)
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

//    //test
//    for(int i = 0; i < g.vCount; i++)
//    {
//        auto &labelCnt = labelCntPerVertice.at(i);
//        auto &maxLabel = maxLabelCnt.at(i);
//
//        for(auto item : labelCnt)
//        {
//            if(maxLabel.second <= item.second)
//            {
//                maxLabel.first = item.first;
//                maxLabel.second = item.second;
//            }
//        }
//    }

    for (int i = 0; i < g.vCount; i++)
    {
        if (maxLabelCnt.at(i).second != 0)
        {
            g.verticesValue.at(i) = LPA_Value(i, maxLabelCnt.at(i).first, maxLabelCnt.at(i).second);
        } else
        {
            g.verticesValue.at(i) = LPA_Value(i, i, 0);
        }
    }
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g,
                                                                    const std::vector<int> &initVSet,
                                                                    std::set<int> &activeVertices)
{
    MessageValueType *mValues = new MessageValueType[this->totalMValuesCount];

    int computeCnt = 0;

    std::vector<ComputeUnitPackage<VertexValueType>> computePackages;

    ComputeUnit<VertexValueType> *computeUnits = nullptr;

    for (int i = 0; i < g.eCount; i++)
    {
        int destVId = g.eList[i].dst;
        int srcVId = g.eList[i].src;

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

    int totalCnt = 0;
    for (auto &computePackage : computePackages)
    {
        computeCnt = computePackage.getCount();
        computeUnits = computePackage.getUnitPtr();

        MSGGenMerge_array(computeCnt, computeUnits, mValues);
        MSGApply_array(computeCnt, computeUnits, mValues);

        for (int i = 0; i < computeCnt; i++, totalCnt++)
        {
            g.verticesValue[totalCnt] = computeUnits[i].destValue;
        }
    }
    free(mValues);
    for (auto &computePackage : computePackages)
    {
        free(computePackage.getUnitPtr());
    }
}

template<typename VertexValueType, typename MessageValueType>
void
LabelPropagation<VertexValueType, MessageValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet<MessageValueType> mGenSet = MessageSet<MessageValueType>();
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while (activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
}


template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g,
                                                                 const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for (int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<MessageValueType>> mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    std::vector<MessageSet<MessageValueType>> mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size(), 1000);

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while (iterCount < 50)
    {
        std::cout << "iterCount: " << iterCount << std::endl;
        auto start = std::chrono::system_clock::now();

        std::cout << "divide graph..." << std::endl;
        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);
        auto divideGraphFinish = std::chrono::system_clock::now();
        for (int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));
        activeVertice.clear();

        auto mergeGraphStart = std::chrono::system_clock::now();

        std::cout << "merge graph..." << std::endl;
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        iterCount++;
        auto end = std::chrono::system_clock::now();

//        for(int i = 0; i < g.vCount; i++)
//            std::cout << i << " " << g.verticesValue.at(i).label << std::endl;
    }

    for (int i = 0; i < g.vCount; i++)
        std::cout << i << " " << g.verticesValue.at(i).label << std::endl;

    Free();
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::download(VertexValueType *vValues, Vertex *vSet,
                                                                   int computeUnitCount,
                                                                   ComputeUnit<VertexValueType> *computeUnits)
{
    for (int i = 0; i < computeUnitCount; i++, pipeDownloadCnt++)
    {
        int destVId = computeUnits[i].destVertex.vertexID;
        int srcVId = computeUnits[i].srcVertex.vertexID;

        vValues[pipeDownloadCnt] = computeUnits[i].destValue;

        vSet[destVId].isActive |= computeUnits[i].destVertex.isActive;
        vSet[srcVId].isActive |= computeUnits[i].srcVertex.isActive;
    }
}

template<typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::IterationEnd(MessageValueType *mValues)
{

}
