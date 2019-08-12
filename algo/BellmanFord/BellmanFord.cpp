//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

template <typename VertexValueType>
BellmanFord<VertexValueType>::BellmanFord()
{
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<VertexValueType> &mSet)
{
    //Reset active vertices info
    for(auto &v : g.vList)
        v.isActive = false;

    for(auto m : mSet.mSet)
    {
        if(g.verticesValue.at(m.dst * this->numOfInitV + g.vList.at(m.src).initVIndex) > m.value)
        {
            g.verticesValue.at(m.dst * this->numOfInitV + g.vList.at(m.src).initVIndex) = m.value;
            activeVertices.insert(m.dst);
            g.vList.at(m.dst).isActive = true;
        }
    }
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<VertexValueType> &mSet)
{
    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.vCount * this->numOfInitV);
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        mSet.insertMsg(Message<VertexValueType>(initVSet.at(i % this->numOfInitV), i / this->numOfInitV, INVALID_MASSAGE));

    for(auto e : g.eList)
    {
        if(g.vList.at(e.src).isActive)
        {
            int vID = e.src;
            if(g.vList.at(vID).vertexID == vID) // It should be of equal value
            {
                for(int i = 0; i < this->numOfInitV; i++)
                {
                    auto &m = mSet.mSet.at(e.dst * this->numOfInitV + i);
                    if(m.value > g.verticesValue.at(e.src * this->numOfInitV + i) + e.weight)
                        m.value = g.verticesValue.at(e.src * this->numOfInitV + i) + e.weight;
                }
            }
            else;
        }
    }
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues)
{
    for(int i = 0; i < vCount; i++) vSet[i].isActive = false;

    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(vValues[i] > mValues[i])
        {
            vValues[i] = mValues[i];
            vSet[i / numOfInitV].isActive = true;
        }
    }
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues)
{
    for(int i = 0; i < vCount * numOfInitV; i++) mValues[i] = INVALID_MASSAGE;

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive)
        {
            for(int j = 0; j < numOfInitV; j++)
            {
                if(mValues[eSet[i].dst * numOfInitV + j] > vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight)
                    mValues[eSet[i].dst * numOfInitV + j] = vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight;
            }
        }
    }
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->numOfInitV = numOfInitV;

    //Memory parameter init
    this->totalVValuesCount = vCount * numOfInitV;
    this->totalMValuesCount = vCount * numOfInitV;
}

template<typename VertexValueType>
void BellmanFord<VertexValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                             const std::vector<int> &initVList)
{
    int numOfInitV_init = initVList.size();

    //v Init
    for(int i = 0; i < numOfInitV_init; i++)
        g.vList.at(initVList.at(i)).initVIndex = i;
    for(auto &v : g.vList)
    {
        if(v.initVIndex != INVALID_INITV_INDEX)
        {
            activeVertices.insert(v.vertexID);
            v.isActive = true;
        }
        else v.isActive = false;
    }

    //vValues init
    g.verticesValue.reserve(g.vCount * numOfInitV_init);
    g.verticesValue.assign(g.vCount * numOfInitV_init, INT32_MAX >> 1);
    for(int initID : initVList)
        g.verticesValue.at(initID * numOfInitV_init + g.vList.at(initID).initVIndex) = 0;
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::Free()
{

}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                const std::vector<int> &initVList)
{
    //Init
    activeVertices.clear();
    for(auto &v : g.vList) v.isActive = false;

    //Merge graphs
    for(const auto &subG : subGSet)
    {
        //vSet merge
        for(int i = 0; i < subG.vCount; i++)
            g.vList.at(i).isActive |= subG.vList.at(i).isActive;

        //vValues merge
        for(int i = 0; i < subG.verticesValue.size(); i++)
        {
            if(g.verticesValue.at(i) > subG.verticesValue.at(i))
                g.verticesValue.at(i) = subG.verticesValue.at(i);
        }
    }

    //Merge active vertices set
    for(const auto &AVs : activeVerticeSet)
    {
        for(auto av : AVs)
            activeVertices.insert(av);
    }
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    MessageSet<VertexValueType> mGenSet = MessageSet<VertexValueType>();
    MessageSet<VertexValueType> mMergedSet = MessageSet<VertexValueType>();

    mMergedSet.mSet.clear();
    MSGGenMerge(g, initVSet, activeVertices, mMergedSet);

    //Test
    std::cout << "MGenMerge:" << clock() << std::endl;
    //Test end

    activeVertices.clear();
    MSGApply(g, initVSet, activeVertices, mMergedSet);

    //Test
    std::cout << "Apply:" << clock() << std::endl;
    //Test end
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();
    MessageSet<VertexValueType> mGenSet = MessageSet<VertexValueType>();
    MessageSet<VertexValueType> mMergedSet = MessageSet<VertexValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while(activeVertices.size() > 0)
        ApplyStep(g, initVList, activeVertices);

    Free();
}

template <typename VertexValueType>
void BellmanFord<VertexValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<VertexValueType>> mGenSetSet = std::vector<MessageSet<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<VertexValueType>());
    std::vector<MessageSet<VertexValueType>> mMergedSetSet = std::vector<MessageSet<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<VertexValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while(activeVertices.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for(int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            AVSet.at(i) = activeVertices;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for(int i = 0; i < partitionCount; i++)
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
