//
// Created by cave-g-f on 2019-05-17.
//

#include "LabelPropagation.h"

#include <iostream>
#include <chrono>

template <typename VertexValueType, typename MessageValueType>
LabelPropagation<VertexValueType, MessageValueType>::LabelPropagation()
{
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if(g.eCount <= 0 || g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType [g.eCount];

    for(int i = 0; i < g.eCount; i++)
    {
        mValues[i] = mSet.mSet.at(i).value;
    }

    //array form computation
    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);

    delete[] mValues;

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if(g.eCount <= 0 || g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType [g.eCount];

    //array form computation
    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);

    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.eCount);

    for(int i = 0; i < g.eCount; i++)
    {
        mSet.insertMsg(Message<MessageValueType>(g.eList.at(i).src, g.eList.at(i).dst, mValues[i]));
    }

    delete[] mValues;

    return g.eCount;
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues)
{
    //store the labels of neighbors for each vertice
    auto labelsVector = std::vector<std::map<int, int>>();

    labelsVector.reserve(vCount);
    labelsVector.assign(vCount, std::map<int, int>());

    for(int i = 0; i < eCount; i++)
    {
        auto &labelCntMap = labelsVector.at(mValues[i].first);

        if(labelCntMap.find(mValues[i].second) == labelCntMap.end())
        {
            labelCntMap[mValues[i].second] = 1;
        }
        else
        {
            labelCntMap.at(mValues[i].second)++;
        }
    }

    //set vValue
    auto valueCnt = 0;
    for(int i = 0; i < vCount; i++)
    {
        auto labelMap = labelsVector.at(i);
        if(!labelMap.empty())
        {
            for(auto &label : labelMap)
            {
                vValues[valueCnt] = LPA_Value(i, label.first, label.second);
                valueCnt++;
            }
        }
    }
    //set the end flag of vValue array
    if(valueCnt < eCount)
    {
        vValues[valueCnt] = LPA_Value(INVALID_INITV_INDEX, -1, -1);
    }

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    for(int i = 0; i < eCount; i++)
    {
        //msg value -- <destinationID, Label>
        auto msgValue = std::pair<int, int>(eSet[i].dst, vValues[eSet[i].src].label);
        mValues[i] = msgValue;
    }

    return eCount;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    //vValue size = e in order to store the label info for merging the subgraph
    this->totalVValuesCount = eCount;
    this->totalMValuesCount = eCount;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    //vValues init
    g.verticesValue.reserve(this->totalVValuesCount);
    g.verticesValue.assign(this->totalVValuesCount, LPA_Value(INVALID_INITV_INDEX, -1, 0));

    for(int i = 0; i < g.vCount; i++)
    {
        g.verticesValue.at(i) = LPA_Value(g.vList.at(i).vertexID, g.vList.at(i).vertexID, 0);
    }
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Free()
{

}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
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

    for(const auto &subG : subGSet)
    {
        //vValues merge
        for(int i = 0; i < subG.eCount; i++)
        {
            auto lpaValue = subG.verticesValue.at(i);

            if(lpaValue.destVId == INVALID_INITV_INDEX)
            {
                break;
            }

            auto &labelCnt = labelCntPerVertice.at(lpaValue.destVId);
            auto &maxLabel = maxLabelCnt.at(lpaValue.destVId);

            if(labelCnt.find(lpaValue.label) == labelCnt.end())
            {
                labelCnt[lpaValue.label] = lpaValue.labelCnt;
            }
            else
            {
                labelCnt[lpaValue.label] += lpaValue.labelCnt;
            }

            if(maxLabel.second <= labelCnt[lpaValue.label])
            {
                maxLabel.first = lpaValue.label;
                maxLabel.second = labelCnt[lpaValue.label];
            }
        }
    }

    for(int i = 0; i < g.vCount; i++)
    {
        if(maxLabelCnt.at(i).second != 0)
        {
            g.verticesValue.at(i) = LPA_Value(i, maxLabelCnt.at(i).first, maxLabelCnt.at(i).second);
        }
        else
        {
            g.verticesValue.at(i) = LPA_Value(i, i, 0);
        }
    }
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    mMergedSet.mSet.clear();

    auto start = std::chrono::system_clock::now();
    MSGGenMerge(g, initVSet, activeVertices, mMergedSet);
    auto mergeEnd = std::chrono::system_clock::now();

    MSGApply(g, initVSet, activeVertices, mMergedSet);
    auto applyEnd = std::chrono::system_clock::now();
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
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
void LabelPropagation<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<MessageValueType>> mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    std::vector<MessageSet<MessageValueType>> mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while(iterCount < 100)
    {
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

//        for(int i = 0; i < g.vCount; i++)
//            std::cout << i << " " << g.verticesValue.at(i).label << std::endl;
    }

    for(int i = 0; i < g.vCount; i++)
        std::cout << i << " " << g.verticesValue.at(i).label << std::endl;

    Free();
}
