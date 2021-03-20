//
// Created by Thoh Testarossa on 2019-04-05.
//

#pragma once

#ifndef GRAPH_ALGO_UTILCLIENT_H
#define GRAPH_ALGO_UTILCLIENT_H

#include "../core/GraphUtil.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../include/UNIX_macro.h"
#include "ThreadPool.h"

template <typename VertexValueType, typename MessageValueType>
class UtilClient
{
public:
    UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo = 0, int pipeVCount = 0, int pipeECount = 0);
    ~UtilClient() = default;

    int connect();
    int transfer(int *initVSet, bool *filteredV, int *timestamp);
    int update(VertexValueType *vValues, Vertex *vSet, int *avSet = nullptr, int avCount = -1);
    int update(VertexValueType *vValues, int *avSet = nullptr, int avCount = -1);
    void requestMSGApply();
    void requestMSGMerge();
    void disconnect();
    void shutdown();
    void graphInit();

    void initPipeline(int threadNum);
    void startPipeline(VertexValueType *vValues, Vertex *vSet, Edge *eSet);
    void stopPipeline();

    int nodeNo;

    int totalVCount;
    int totalECount;
    int pipeVCount;
    int pipeECount;
    int numOfInitV;

    int *initVSet;
    bool *filteredV;
    int *timestamp;
    MessageValueType *mValues;

    VertexValueType *vValuesUpdate;
    VertexValueType *vValuesDownload;
    VertexValueType *vValuesCompute;
    Vertex *vSetUpdate;
    Vertex *vSetDownload;
    Vertex *vSetCompute;
    Edge *eSetUpdate;
    Edge *eSetDownload;
    Edge *eSetCompute;

    int *avSet;
    int *avCount;

private:
    UNIX_shm initVSet_shm;
    UNIX_shm filteredV_shm;
    UNIX_shm timestamp_shm;
    UNIX_shm mValues_shm;
    UNIX_shm avSet_shm;
    UNIX_shm avCount_shm;

    UNIX_shm vValuesUpdate_shm;
    UNIX_shm vValuesDownload_shm;
    UNIX_shm vValuesCompute_shm;
    UNIX_shm vSetUpdate_shm;
    UNIX_shm vSetDownload_shm;
    UNIX_shm vSetCompute_shm;
    UNIX_shm eSetUpdate_shm;
    UNIX_shm eSetDownload_shm;
    UNIX_shm eSetCompute_shm;

    UNIX_msg server_msq;
    UNIX_msg client_msq;

    std::shared_ptr<ThreadPool> threadPoolPtr = nullptr;
};

#endif //GRAPH_ALGO_UTILCLIENT_H
