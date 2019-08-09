//
// Created by Thoh Testarossa on 2019-08-08.
//

#include "ConnectedComponent.h"

#include <iostream>
#include <ctime>

template<typename VertexValueType>
ConnectedComponent<VertexValueType>::ConnectedComponent()
{

}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                   std::set<int> &activeVertice,
                                                   const MessageSet<VertexValueType> &mSet)
{
    //Availability check
    if(g.vCount <= 0) return;

    VertexValueType *mValues = new VertexValueType [g.vCount];
    for(int i = 0; i < g.vCount; i++) mValues[i] = INVALID_MASSAGE;
    for(const auto &m : mSet.mSet) mValues[m.dst] = m.value;

    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], 0, nullptr, &g.verticesValue[0], mValues);

    for(const auto &v : g.vList)
    {
        if(v.isActive)
            activeVertice.insert(v.vertexID);
    }
}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                      const std::set<int> &activeVertice,
                                                      MessageSet<VertexValueType> &mSet)
{
    //Availability check
    if(g.vCount <= 0) return;

    VertexValueType *mValues = new VertexValueType [g.vCount];

    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, nullptr, &g.verticesValue[0], mValues);

    //Package mValues into result mSet
    for(int i = 0; i < g.vCount; i++)
    {
        if(mValues[i] != (VertexValueType)INVALID_MASSAGE)
            mSet.insertMsg(Message<VertexValueType>(INVALID_INITV_INDEX, i, mValues[i]));
    }
}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV,
                                                         const int *initVSet, VertexValueType *vValues,
                                                         VertexValueType *mValues)
{
    //Activity reset
    for(int i = 0; i < vCount; i++) vSet[i].isActive = false;

    //vValue apply
    for(int i = 0; i < vCount; i++)
    {
        if(vValues[i] > mValues[i])
        {
            vValues[i] = mValues[i];
            vSet[i].isActive = true;
        }
    }
}

template<typename VertexValueType>
void
ConnectedComponent<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet,
                                                       int numOfInitV, const int *initVSet,
                                                       const VertexValueType *vValues, VertexValueType *mValues)
{
    for(int i = 0; i < vCount; i++) mValues[i] = (VertexValueType)INVALID_MASSAGE;

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive)
        {
            if(mValues[eSet[i].dst] < vValues[eSet[i].src])
                mValues[eSet[i].dst] = vValues[eSet[i].src];
        }
    }
}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::MergeGraph(Graph<VertexValueType> &g,
                                                     const std::vector<Graph<VertexValueType>> &subGSet,
                                                     std::set<int> &activeVertices,
                                                     const std::vector<std::set<int>> &activeVerticeSet,
                                                     const std::vector<int> &initVList)
{
    //Init
    activeVertices.clear();
    for(auto &v : g.vList) v.isActive = false;

    for(const auto &subG : subGSet)
    {
        //vSet merge
        for(int i = 0; i < subG.vCount; i++)
            g.vList.at(i).isActive |= subG.vList.at(i).isActive;

        //vValues merge
        for(int i = 0; i < subG.vCount; i++)
        {
            if(g.verticesValue.at(i) > subG.verticesValue.at(i))
                g.verticesValue.at(i) = subG.verticesValue.at(i);
        }
    }

    for(const auto &AVs : activeVerticeSet)
    {
        for(auto av : AVs)
            activeVertices.insert(av);
    }
}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::MergeMergedMSG(MessageSet<VertexValueType> &mergedMSG,
                                                         const std::vector<MessageSet<VertexValueType>> &mergedMSGSet)
{

}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->totalVValuesCount = vCount;
    this->totalMValuesCount = vCount;
}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                    const std::vector<int> &initVList)
{
    //v Init
    for(auto &v : g.vList) v.isActive = true;

    //vValues init
    g.verticesValue.reserve(g.vCount);
    for(int i = 0; i < g.vCount; i++) g.verticesValue.at(i) = (VertexValueType)i;
}


template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::Deploy(int vCount, int numOfInitV)
{

}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::Free()
{

}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                    std::set<int> &activeVertices)
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

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{

}

template<typename VertexValueType>
void ConnectedComponent<VertexValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList,
                                                 int partitionCount)
{

}
