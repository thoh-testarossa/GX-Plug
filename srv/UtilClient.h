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
    UtilClient(int numOfInitV, int nodeNo = 0, int threadNum = 0);
    ~UtilClient() = default;

    int connect();
    int transfer(VertexValueType *vValues, Vertex *vSet);
    void requestMSGApply();
    void requestMSGMerge();
    void disconnect();
    void shutdown();
    void graphInit();

    void initPipeline(int threadNum);
    void startPipeline(ComputeUnitPackage<VertexValueType> *computePackages, int packagesCnt);
    void stopPipeline();

    int nodeNo;

    int numOfInitV;

    int *initVSet;
    bool *filteredV;
    int *timestamp;
    MessageValueType *mValues;

    ComputeUnit<VertexValueType> *computeUnitsUpdate;
    ComputeUnit<VertexValueType> *computeUnitsCompute;
    ComputeUnit<VertexValueType> *computeUnitsDownload;
    int *updateCnt;
    int *computeCnt;
    int *downloadCnt;

    int *avSet;
    int *avCount;

private:

    void rotate();
    int update(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits);
    int download();

    UNIX_shm initVSet_shm;
    UNIX_shm filteredV_shm;
    UNIX_shm timestamp_shm;
    UNIX_shm mValues_shm;
    UNIX_shm avSet_shm;
    UNIX_shm avCount_shm;

    UNIX_shm computeUnitsUpdate_shm;
    UNIX_shm computeUnitsCompute_shm;
    UNIX_shm computeUnitsDownload_shm;
    UNIX_shm updateCnt_shm;
    UNIX_shm computeCnt_shm;
    UNIX_shm downloadCnt_shm;

    UNIX_msg server_msq;
    UNIX_msg client_msq;

    VertexValueType *vValues_agent;
    Vertex *vSets_agent;

    std::shared_ptr<ThreadPool> threadPoolPtr = nullptr;
};

#endif //GRAPH_ALGO_UTILCLIENT_H
