//
// Created by Thoh Testarossa on 2019-08-22.
//

#include "DDFS.h"

#include <iostream>
#include <ctime>

template <typename VertexValueType, typename MessageValueType>
DDFS<VertexValueType, MessageValueType>::DDFS()
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                     std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                        const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet,
                                           VertexValueType *vValues, MessageValueType *mValues)
{
    //Reset vertex activity
    for(int i = 0; i < vCount; i++)
        vSet[i].isActive = false;

    //Reset opbit & vNextMSGTo
    for(int i = 0; i < vCount; i++)
    {
        vValues[i].opbit = 0;
        vValues[i].vNextMSGTo = -1;
    }

    //Check each msgs
    //msgs are sent from edges
    //eCount here is the account of edges which contains messages rather than the account of g's edges
    for(int i = 0; i < eCount; i++)
    {
        //msg token check
        if(mValues[i].msgbit & MSG_TOKEN)
        {
            if(vValues[mValues[i].dst].state == STATE_IDLE)
            {
                //Mark j as i's father
                //There should be some approach more efficient
                for(auto &vState : vValues[mValues[i].dst].vStateList)
                {
                    if(vState.first == mValues[i].src)
                    {
                        vState.second = MARK_PARENT;
                        break;
                    }
                }

                vValues[mValues[i].dst].state = STATE_DISCOVERED;
                vValues[mValues[i].dst].vNextMSGTo = this->search(mValues[i].dst, numOfInitV, initVSet, vSet, vValues);

                //prepare to broadcast msg "visited" to other vertices
                vValues[mValues[i].dst].opbit |= OP_BROADCAST;

                //Vertex which will send msg will be activated
                vSet[mValues[i].dst].isActive = true;
            }
        }
        //msg visited check
        else if(mValues[i].msgbit & MSG_VISITED)
        {
            //There should be some approach more efficient
            for(auto &vState : vValues[mValues[i].dst].vStateList)
            {
                if(vState.first == mValues[i].src)
                {
                    if(vState.second == MARK_UNVISITED)
                    {
                        vState.second = MARK_VISITED;
                        vValues[mValues[i].dst].vNextMSGTo = -1;
                    }
                    else if(vState.second == MARK_SON)
                    {
                        vState.second = MARK_VISITED;
                        vValues[mValues[i].dst].vNextMSGTo = this->search(mValues[i].dst, numOfInitV, initVSet, vSet, vValues);
                    }
                }
            }
        }
        else;
    }
}

template <typename VertexValueType, typename MessageValueType>
void
DDFS<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV,
                                         const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    int msgCount = 0;

    for(int i = 0; i < vCount; i++)
    {
        //Check if needed to generate broadcast msg
        if(vValues[i].opbit & OP_BROADCAST)
        {
            for(const auto &vState : vValues[i].vStateList)
            {
                if(vState.second == MARK_UNVISITED || vState.second == MARK_VISITED)
                {
                    mValues[msgCount].src = i;
                    mValues[msgCount].dst = vState.first;
                    mValues[msgCount].msgbit = MSG_VISITED;
                    //Not implemented yet
                    mValues[msgCount].timestamp = 0;

                    msgCount++;
                }
            }
        }
        //Check if needed to generate search msg
        if(vValues[i].opbit & OP_MSG_FROM_SEARCH)
        {
            mValues[msgCount].src = i;
            mValues[msgCount].dst = vValues[i].vNextMSGTo;
            mValues[msgCount].msgbit = MSG_TOKEN;
            //Not implemented yet
            mValues[msgCount].timestamp = 0;

            msgCount++;
        }
    }
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                                       std::set<int> &activeVertices,
                                       const std::vector<std::set<int>> &activeVerticeSet,
                                       const std::vector<int> &initVList)
{
    //Reset global vValues
    for(auto &vV : g.verticesValue)
    {

    }

    //Merge subGs
    for(const auto &subG : subGSet)
    {
        //state merge
        //opbit merge
        //vNextMSGTo merge
        //startTime merge
        //endTime merge
        //relatedVCount merge
        //vStateList merge
    }
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                      const std::vector<int> &initVList)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Free()
{

}

template <typename VertexValueType, typename MessageValueType>
int DDFS<VertexValueType, MessageValueType>::search(int vid, int numOfInitV, const int *initVSet, Vertex *vSet, VertexValueType *vValues)
{
    bool chk = false;
    for(auto &vState : vValues[vid].vStateList)
    {
        if(!(vState.second == MARK_VISITED))
        {
            chk = true;
            vState.second = MARK_SON;
            vValues[vid].opbit |= OP_MSG_FROM_SEARCH;
            vValues[vid].opbit |= OP_MSG_DOWNWARD;
            //Vertex which will send msg will be activated
            vSet[vid].isActive = true;
            return vState.first;
        }
    }

    if(!chk)
    {
        if(vid == initVSet[0]) return -1;
        else
        {
            //There should be some approach more efficient
            for(auto &vState : vValues[vid].vStateList)
            {
                if(vState.second == MARK_PARENT)
                {
                    //Vertex which will send msg will be activated
                    vSet[vid].isActive = true;
                    vValues[vid].opbit |= OP_MSG_FROM_SEARCH;
                    return vState.first;
                }
            }

        }
    }

    return 0;
}

