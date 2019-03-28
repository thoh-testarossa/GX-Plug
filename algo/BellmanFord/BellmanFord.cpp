//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

BellmanFord::BellmanFord()
{
}

void BellmanFord::MSGApply(Graph &g, std::set<int> &activeVertice, const MessageSet &mSet)
{
    //Reset active vertices info
    for(auto &v : g.vList)
        v.isActive = false;

    for(auto m : mSet.mSet)
    {
        if(g.vList.at(m.dst).value.find(m.src)->second > m.value)
        {
            g.vList.at(m.dst).value.find(m.src)->second = m.value;
            activeVertice.insert(m.dst);
            g.vList.at(m.dst).isActive = true;
        }
    }
}

void BellmanFord::MSGGen(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet)
{
    for(auto e : g.eList)
    {
        if(g.vList.at(e.src).isActive)
        {
            int vID = e.src;
            if(g.vList.at(vID).vertexID == vID) // It should be of equal value
            {
                for(auto vV : g.vList.at(vID).value)
                    mSet.insertMsg(Message(vV.first, e.dst, vV.second + e.weight));
            }
            else;
        }
    }
}

void BellmanFord::MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source)
{
    std::map<int, MessageSet> mergeMap = std::map<int, MessageSet>();

    for(auto m : source.mSet)
    {
        if(mergeMap.find(m.src) == mergeMap.end()) //result didn't contain any message about m's src
        {
            mergeMap.insert(std::pair<int, MessageSet>(m.src, MessageSet()));
            mergeMap.find(m.src)->second.insertMsg(m);
        }
        else
        {
            bool isContained = false;

            for(auto &m2 : mergeMap.find(m.src)->second.mSet) //result contains messages about m's src
            {
                if(m2.dst == m.dst) //result contains messages from m.src to m.dst
                {
                    if(m2.value > m.value) m2.value = m.value; //The messages in result should be updated
                    isContained = true;
                    break;
                }
            }

            if(!isContained) //result didn't contain messages from m.src to m.dst
                mergeMap.find(m.src)->second.insertMsg(m);
        }
    }

    for(auto mSetPair : mergeMap)
    {
        for(auto m : mSetPair.second.mSet)
            result.insertMsg(m);
    }
}

void BellmanFord::MSGGenMerge(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet)
{
    //Old way to do this two step
    MSGGen(g, activeVertice, mSet);
    auto mSetBeforeMerged = mSet;
    mSet.mSet.clear();
    MSGMerge(g, mSet, mSetBeforeMerged);

    //Generate merged msgs directly

}

void BellmanFord::Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
{
    this->numOfInitV = initVList.size();

    for(auto &v : g.vList)
    {
        for(auto iV : initVList)
        {
            if (v.vertexID == iV)
            {
                v.value.insert(std::pair<int, double>(iV, 0));
                activeVertice.insert(v.vertexID);
                v.isActive = true;
            }
            else
                v.value.insert(std::pair<int, double>(iV, INT32_MAX >> 1));
        }
    }
}

void BellmanFord::Deploy(Graph &g, int numOfInitV)
{
    this->numOfInitV = numOfInitV;
}

void BellmanFord::Free(Graph &g)
{

}

void BellmanFord::MergeGraph(Graph &g, const std::vector<Graph> &subGSet,
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

        for(auto subG : subGSet)
        {
            //Merge vertices info
            for(auto v : subG.vList)
            {
                if(v.isActive)
                    resG.vList.at(v.vertexID).isActive = true;

                for(auto vV : v.value)
                {
                    if(resG.vList.at(v.vertexID).value.find(vV.first)->second > vV.second)
                        resG.vList.at(v.vertexID).value.find(vV.first)->second = vV.second;
                }
            }

            //Merge Edge
            //There should be not any relevant edges since subgraphs are divided by dividing edge set
            for(auto e : subG.eList)
                resG.insertEdge(e.src, e.dst, e.weight);
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

void BellmanFord::MergeMergedMSG(MessageSet &mergedMSG, const std::vector<MessageSet> &mergedMSGSet)
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
        mergedMSG.insertMsg(Message(m.first.first, m.first.second, m.second));
}

void BellmanFord::ApplyStep(Graph &g, std::set<int> &activeVertice)
{
    MessageSet mGenSet = MessageSet();
    MessageSet mMergedSet = MessageSet();

    //mGenSet.mSet.clear();
    //MSGGen(g, activeVertice, mGenSet);

    //Test
    //std::cout << "Gen:" << clock() << std::endl;
    //Test end

    //Test
    /*
    for(int j = 0; j < mGenSet.mSet.size(); j++)
        mGenSet.mSet.at(j).print();
    */
    //Test end

    //mMergedSet.mSet.clear();
    //MSGMerge(g, mMergedSet, mGenSet);

    //Test
    //std::cout << "MMerge:" << clock() << std::endl;
    //Test end

    //Test
    /*
    for(int j = 0; j < mMergedSet.mSet.size(); j++)
        mMergedSet.mSet.at(j).print();
    std::cout << std::endl;
    std::cout << "######################################" << std::endl;
    */
    //Test end

    mMergedSet.mSet.clear();
    MSGGenMerge(g, activeVertice, mMergedSet);

    //Test
    std::cout << "MGenMerge:" << clock() << std::endl;
    //Test end

    //Test
    /*
    for(int j = 0; j < mMergedSet.mSet.size(); j++)
        mMergedSet.mSet.at(j).print();
    std::cout << std::endl;
    std::cout << "######################################" << std::endl;
    */
    //Test end

    activeVertice.clear();
    MSGApply(g, activeVertice, mMergedSet);

    //Test
    std::cout << "Apply:" << clock() << std::endl;
    //Test end

    //Test
    /*
    for(int j = 0; j < g.vCount; j++)
    {
        for(auto iter = g.vList.at(j).value.begin(); iter != g.vList.at(j).value.end(); iter++)
            std::cout << iter->second << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    //Test end
}

void BellmanFord::Apply(Graph &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet mGenSet = MessageSet();
    MessageSet mMergedSet = MessageSet();

    Init(g, activeVertice, initVList);

    Deploy(g, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, activeVertice);

    Free(g);
}

void BellmanFord::ApplyD(Graph &g, const std::vector<int> &initVList, int partitionCount)
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

    Deploy(g, initVList.size());

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
        /*
        for(int i = 0; i < partitionCount; i++)
        {
            for(int j = 0; j < subGraphSet.at(i).vCount; j++)
            {
                for(auto iter = subGraphSet.at(i).vList.at(j).value.begin(); iter != subGraphSet.at(i).vList.at(j).value.end(); iter++)
                    std::cout << iter->second << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "######################################" << std::endl;
        */
        //Test end

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), AVSet.at(i));


        /*
        for(int i = 0; i < partitionCount; i++)
        {
            mGenSetSet.at(i).mSet.clear();
            MSGGen(subGraphSet.at(i), activeVertice, mGenSetSet.at(i));
        }



        for(int i = 0; i < partitionCount; i++)
        {
            mMergedSetSet.at(i).mSet.clear();
            MSGMerge(g, mMergedSetSet.at(i), mGenSetSet.at(i));
        }



        */

        /*
        auto mMergedSet = MessageSet();
        MergeMergedMSG(mMergedSet, mMergedSetSet);
        for(int i = 0; i < partitionCount; i++) mMergedSetSet.at(i).mSet.clear();
        for(int i = 0; i < partitionCount; i++)
        {
            for(int k = i * mMergedSet.mSet.size() / partitionCount; k < (i + 1) * mMergedSet.mSet.size() / partitionCount; k++)
                mMergedSetSet.at(i).insertMsg(mMergedSet.mSet.at(k));
        }
        */

        /*
        for(int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            MSGApply(subGraphSet.at(i), AVSet.at(i), mMergedSetSet.at(i));
        }


        */

        activeVertice.clear();
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end

        //Test
        /*
        for(int j = 0; j < g.vCount; j++)
        {
            for(auto iter = g.vList.at(j).value.begin(); iter != g.vList.at(j).value.end(); iter++)
                std::cout << iter->second << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "######################################" << std::endl;
        */
        //Test end
    }

    Free(g);

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}
