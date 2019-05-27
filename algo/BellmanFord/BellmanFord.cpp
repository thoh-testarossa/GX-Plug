//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

template <typename T>
BellmanFord<T>::BellmanFord()
{
}

template <typename T>
void BellmanFord<T>::MSGApply(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<T> &mSet)
{
    //Reset active vertices info
    for(auto &v : g.vList)
        v.isActive = false;

    for(auto m : mSet.mSet)
    {
        if(g.verticeValue.at(m.dst * this->numOfInitV + g.vList.at(m.src).initVIndex) > m.value)
        {
            g.verticeValue.at(m.dst * this->numOfInitV + g.vList.at(m.src).initVIndex) = m.value;
            activeVertice.insert(m.dst);
            g.vList.at(m.dst).isActive = true;
        }
    }
}

template <typename T>
void BellmanFord<T>::MSGGenMerge(const Graph<T> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<T> &mSet)
{
    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.vCount * this->numOfInitV);
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        mSet.insertMsg(Message<T>(initVSet.at(i % this->numOfInitV), i / this->numOfInitV, INVALID_MASSAGE));

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
                    if(m.value > g.verticeValue.at(e.src * this->numOfInitV + i) + e.weight)
                        m.value = g.verticeValue.at(e.src * this->numOfInitV + i) + e.weight;
                }
            }
            else;
        }
    }
}

template <typename T>
void BellmanFord<T>::MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, T *vValues, T *mValues)
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

template <typename T>
void BellmanFord<T>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const T *vValues, T *mValues)
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

template <typename T>
void BellmanFord<T>::Init(Graph<T> &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
{
    int numOfinitV_init = initVList.size();

    //v Init
    for(int i = 0; i < numOfinitV_init; i++)
        g.vList.at(initVList.at(i)).initVIndex = i;
    for(auto &v : g.vList)
    {
        if(v.initVIndex != INVALID_INITV_INDEX)
        {
            activeVertice.insert(v.vertexID);
            v.isActive = true;
        }
        else v.isActive = false;
    }

    //vValues init
    g.verticeValue.reserve(g.vCount * numOfinitV_init);
    g.verticeValue.assign(g.vCount * numOfinitV_init, INT32_MAX >> 1);
    for(int initID : initVList)
        g.verticeValue.at(initID * numOfinitV_init + g.vList.at(initID).initVIndex) = 0;
}

template <typename T>
void BellmanFord<T>::Deploy(int vCount, int numOfInitV)
{
    this->numOfInitV = numOfInitV;
}

template <typename T>
void BellmanFord<T>::Free()
{

}

template <typename T>
void BellmanFord<T>::MergeGraph(Graph<T> &g, const std::vector<Graph<T>> &subGSet,
                std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                const std::vector<int> &initVList)
{
    //Merge graphs
    auto resG = Graph<T>(0);

    if(subGSet.size() <= 0);
    else
    {
        resG = subGSet.at(0);

        resG.eList.clear();
        resG.eCount = 0;

        for(const auto &subG : subGSet) resG.eCount += subG.eCount;
        resG.eList.reserve(resG.eCount);

        //Merge subGraphs
        for(const auto &subG : subGSet)
        {
            //Merge vertices info
            for(const auto &v : subG.vList) resG.vList.at(v.vertexID).isActive |= v.isActive;

            //Merge vValues
            for(int i = 0; i < subG.verticeValue.size(); i++)
            {
                if(resG.verticeValue.at(i) > subG.verticeValue.at(i))
                    resG.verticeValue.at(i) = subG.verticeValue.at(i);
            }

            //Merge Edge
            //There should be not any relevant edges since subgraphs are divided by dividing edge set
            resG.eList.insert(resG.eList.end(), subG.eList.begin(), subG.eList.end());
        }
    }

    g = resG;

    //Merge active vertices set
    for(auto AVs : activeVerticeSet)
    {
        for(auto av : AVs)
            activeVertice.insert(av);
    }
}

template <typename T>
void BellmanFord<T>::MergeMergedMSG(MessageSet<T> &mergedMSG, const std::vector<MessageSet<T>> &mergedMSGSet)
{
    auto mergeMap = std::map<std::pair<int, int>, double>();
    for(auto mMSet : mergedMSGSet)
    {
        for(auto mM : mMSet.mSet)
        {
            auto index = std::pair<int, int>(mM.src, mM.dst);
            if (mergeMap.find(index) == mergeMap.end())
                mergeMap.insert(std::pair<std::pair<int, int>, double>(index, mM.value));
            else
            {
                if(mergeMap.find(index)->second > mM.value)
                    mergeMap.find(index)->second = mM.value;
            }
        }
    }

    for(auto m : mergeMap)
        mergedMSG.insertMsg(Message<T>(m.first.first, m.first.second, m.second));
}

template <typename T>
void BellmanFord<T>::ApplyStep(Graph<T> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice)
{
    MessageSet<T> mGenSet = MessageSet<T>();
    MessageSet<T> mMergedSet = MessageSet<T>();

    mMergedSet.mSet.clear();
    MSGGenMerge(g, initVSet, activeVertice, mMergedSet);

    //Test
    std::cout << "MGenMerge:" << clock() << std::endl;
    //Test end

    activeVertice.clear();
    MSGApply(g, initVSet, activeVertice, mMergedSet);

    //Test
    std::cout << "Apply:" << clock() << std::endl;
    //Test end
}

template <typename T>
void BellmanFord<T>::Apply(Graph<T> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet<T> mGenSet = MessageSet<T>();
    MessageSet<T> mMergedSet = MessageSet<T>();

    Init(g, activeVertice, initVList);

    Deploy(g.vCount, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
}

template <typename T>
void BellmanFord<T>::ApplyD(Graph<T> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<T>> mGenSetSet = std::vector<MessageSet<T>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<T>());
    std::vector<MessageSet<T>> mMergedSetSet = std::vector<MessageSet<T>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<T>());

    Init(g, activeVertice, initVList);

    Deploy(g.vCount, initVList.size());

    int iterCount = 0;

    while(activeVertice.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for(int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            AVSet.at(i) = activeVertice;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertice.clear();
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end
    }

    Free();

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}
