//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

BellmanFord::BellmanFord()
{
}

void BellmanFord::MSGApply(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet &mSet)
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

/*
void BellmanFord::MSGGen(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet)
{
    for(auto e : g.eList)
    {
        if(g.vList.at(e.src).isActive)
        {
            int vID = e.src;
            if(g.vList.at(vID).vertexID == vID) // It should be of equal value
            {
                for(int i = vID * this->numOfInitV; i < (vID + 1) * this->numOfInitV; i++)
                    mSet.insertMsg(Message(initVSet.at(i % this->numOfInitV), e.dst, g.verticeValue.at(i) + e.weight));
            }
            else;
        }
    }
}
*/

/*
void BellmanFord::MSGMerge(const Graph &g, const std::vector<int> &initVSet, MessageSet &result, const MessageSet &source)
{
    result.mSet.clear();
    result.mSet.reserve(g.vCount * this->numOfInitV);
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        result.insertMsg(Message(initVSet.at(i % this->numOfInitV), i / this->numOfInitV, INVALID_MASSAGE));

    for(int i = 0; i < source.mSet.size(); i++)
    {
        int _src = source.mSet.at(i).src, _dst = source.mSet.at(i).dst;
        double _value = source.mSet.at(i).value;
        auto &m = result.mSet.at(_dst * this->numOfInitV + g.vList.at(_src).initVIndex);
        if(m.src == _src && m.dst == _dst) //It should be equal
        {
            if(m.value > _value)
                m.value = _value;
        }
    }
}
*/

void BellmanFord::MSGGenMerge(const Graph &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet &mSet)
{
    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.vCount * this->numOfInitV);
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        mSet.insertMsg(Message(initVSet.at(i % this->numOfInitV), i / this->numOfInitV, INVALID_MASSAGE));

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

void BellmanFord::MSGApply_array(int vCount, int numOfInitV, const int *initVSet, bool *AVCheckSet, double *vValues, double *mValues, int *initVIndexSet)
{
    for(int i = 0; i < vCount; i++) AVCheckSet[i] = false;

    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(vValues[i] > mValues[i])
        {
            vValues[i] = mValues[i];
            AVCheckSet[i / numOfInitV] = true;
        }
    }
}

/*
void BellmanFord::MSGGen_array(int vCount, int eCount, int numOfInitV, const int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, int &numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, bool *AVCheckSet)
{
    numOfMSG = 0;

    for(int i = 0; i < eCount; i++)
    {
        if(AVCheckSet[eSrcSet[i]])
        {
            for(int j = 0; j < numOfInitV; j++)
            {
                mInitVSet[numOfMSG] = initVSet[j];
                mDstSet[numOfMSG] = eDstSet[j];
                mValueSet[numOfMSG] = vValues[eSrcSet[i] * numOfInitV + j] + eWeightSet[j];
                numOfMSG++;
            }
        }
    }
}
*/

/*
void BellmanFord::MSGMerge_array(int vCount, int numOfInitV, const int *initVSet, int numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, double *mValues, int *initVIndexSet)
{
    for(int i = 0; i < vCount * numOfInitV; i++) mValues[i] = INVALID_MASSAGE;

    for(int i = 0; i < numOfMSG; i++)
    {
        if(mValues[mDstSet[i] * numOfInitV + initVIndexSet[mInitVSet[i]]] > mValueSet[i])
            mValues[mDstSet[i] * numOfInitV + initVIndexSet[mInitVSet[i]]] = mValueSet[i];
    }
}
*/

void BellmanFord::MSGGenMerge_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, Edge *eSet, double *mValues, bool *AVCheckSet)
{
    for(int i = 0; i < vCount * numOfInitV; i++) mValues[i] = INVALID_MASSAGE;

    for(int i = 0; i < eCount; i++)
    {
        if(AVCheckSet[eSet[i].src])
        {
            for(int j = 0; j < numOfInitV; j++)
            {
                if(mValues[eSet[i].dst * numOfInitV + j] > vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight)
                    mValues[eSet[i].dst * numOfInitV + j] = vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight;
            }
        }
    }
}

void BellmanFord::Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
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

void BellmanFord::Deploy(int vCount, int numOfInitV)
{
    this->numOfInitV = numOfInitV;
}

void BellmanFord::Free()
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

        for(const auto &subG : subGSet) resG.eCount += subG.eCount;
        resG.eList.reserve(resG.eCount);

        //Merge subGraphs
        for(const auto &subG : subGSet)
        {
            //Merge vertices info
            for(auto v : subG.vList) resG.vList.at(v.vertexID).isActive |= v.isActive;

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

void BellmanFord::ApplyStep(Graph &g, const std::vector<int> &initVSet, std::set<int> &activeVertice)
{
    MessageSet mGenSet = MessageSet();
    MessageSet mMergedSet = MessageSet();

    mMergedSet.mSet.clear();
    MSGGenMerge(g, initVSet, activeVertice, mMergedSet);

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
    MSGApply(g, initVSet, activeVertice, mMergedSet);

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

    Deploy(g.vCount, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
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

    Deploy(g.vCount, initVList.size());

    int iterCount = 0;

    while(activeVertice.size() > 0)
    {
        /*
        //Test
        for(auto i : activeVertice) std::cout << i << " ";
        std::cout << std::endl;
        //Test end
        */

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
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));


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

    Free();

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}
