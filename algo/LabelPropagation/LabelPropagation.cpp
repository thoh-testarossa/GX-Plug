//
// Created by cave-g-f on 2019-05-17.
//

#include "LabelPropagation.h"

#include <iostream>
#include <ctime>

LabelPropagation::LabelPropagation()
{
}

void LabelPropagation::MSGApply(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet &mSet)
{
    //Reset active vertices info
    for(auto &v : g.vList)
        v.isActive = true;

    //store the labels of neighbors for each vertice
    auto labelsVector = std::vector<std::map<int, int>>();

    //store the most often label for each vertice
    auto maxCntLabel = std::vector<std::pair<int, int>>();

    labelsVector.reserve(g.vCount);
    labelsVector.assign(g.vCount, std::map<int, int>());
    maxCntLabel.reserve(g.vCount);
    maxCntLabel.assign(g.vCount, std::pair<int, int>(0, 0));

    for(auto m : mSet.mSet)
    {
        auto labelCntMap = labelsVector.at(m.dst);
        auto maxLabel = maxCntLabel.at(m.dst);
        
        if(labelCntMap.find(m.value) == labelCntMap.end())
        {
            labelCntMap.insert(std::map<int, int>::value_type(m.value, 1));
        }
        else
        {
            labelCntMap.at(m.value)++;
        }
        
        if(maxLabel.second < labelCntMap.at(m.value))
        {
            maxLabel.first = m.value;
            maxLabel.second = labelCntMap.at(m.value);
        }
    }

    for(int i = 0; i < maxCntLabel.size(); i++)
    {
        auto label = maxCntLabel.at(i);

        if(label.second != 0)
        {
            g.verticeValue.at(i) = label.first;
            g.verticeLabelCnt.at(i) = label.second;
        }
    }
    
}

void LabelPropagation::MSGGenMerge(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet)
{
    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.eCount);

    for(auto e : g.eList)
    {
        mSet.insertMsg(Message(e.src, e.dst, g.verticeValue.at(e.src)));
    }
    
}

void LabelPropagation::MSGApply_array(int vCount, Vertex *vSet, int numOfInitV, const int *initVSet, double *vValues, double *mValues)
{
    for(int i = 0; i < vCount; i++)
    {
        vSet[i].isActive = true; 
    }
}

void LabelPropagation::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const double *vValues, double *mValues)
{
    for(int i = 0; i < eCount; i++)
    {
        mValues[i] = vSet[eSet[i].src].vertexID;
    }
}

void LabelPropagation::Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
{
    //vValues init
    g.verticeValue.reserve(g.vCount);
    g.verticeValue.assign(g.vCount, -1);
    g.verticeLabelCnt.reserve(g.vCount);
    g.verticeLabelCnt.assign(g.vCount, 0);

    for(int i = 0; i < g.vList.size(); i++)
    {
        g.verticeValue.at(i) = g.vList.at(i).vertexID;
        g.verticeLabelCnt.at(i) = 1;
    }
}

void LabelPropagation::Deploy(int vCount, int numOfInitV)
{

}

void LabelPropagation::Free()
{

}

void LabelPropagation::MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
                std::set<int> &activeVertice, const std::vector<std::set<int>> &activeVerticeSet,
                const std::vector<int> &initVList)
{
    //Merge graphs
    auto resG = Graph(0);

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
                if(resG.verticeValue.at(i) != subG.verticeValue.at(i) && resG.verticeLabelCnt.at(i) < subG.verticeLabelCnt.at(i))
                {
                    resG.verticeLabelCnt.at(i) = subG.verticeLabelCnt.at(i);
                    resG.verticeValue.at(i) = subG.verticeValue.at(i);
                }
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

void LabelPropagation::MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet)
{
    auto mergeMap = std::map<std::pair<int, int>, double>();
    for(auto mMSet : mergedMSGSet)
    {
        for(auto mM : mMSet.mSet)
        {
            auto index = std::pair<int, int>(mM.src, mM.dst);
            if (mergeMap.find(index) == mergeMap.end())
            {
                mergeMap.insert(std::pair<std::pair<int, int>, double>(index, mM.value));
            }
        }
    }

    for(auto m : mergeMap)
        mergedMSG.insertMsg(Message(m.first.first, m.first.second, m.second));
}

void LabelPropagation::ApplyStep(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice)
{
    MessageSet mGenSet = MessageSet();
    MessageSet mMergedSet = MessageSet();

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

void LabelPropagation::Apply(Graph &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet mGenSet = MessageSet();
    MessageSet mMergedSet = MessageSet();

    Init(g, activeVertice, initVList);

    Deploy(g.vCount, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
}

void LabelPropagation::ApplyD(Graph &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet> mGenSetSet = std::vector<MessageSet>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet());
    std::vector<MessageSet> mMergedSetSet = std::vector<MessageSet>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet());

    Init(g, activeVertice, initVList);

    Deploy(g.vCount, initVList.size());

    int iterCount = 0;

    while(activeVertice.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = DivideGraphByEdge(g, partitionCount);

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

