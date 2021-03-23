//
// Created by Thoh Testarossa on 2019-04-05.
//

#include <cstring>
#include <iostream>
#include <memory>
#include "UtilClient.h"

template<typename VertexValueType, typename MessageValueType>
UtilClient<VertexValueType, MessageValueType>::UtilClient(int numOfInitV, int nodeNo, int threadNum)
{
    this->nodeNo = nodeNo;

    this->numOfInitV = numOfInitV;

    this->mValues_shm = UNIX_shm();
    this->initVSet_shm = UNIX_shm();
    this->filteredV_shm = UNIX_shm();
    this->timestamp_shm = UNIX_shm();
    this->avCount_shm = UNIX_shm();
    this->avSet_shm = UNIX_shm();

    this->computeUnitsUpdate_shm = UNIX_shm();
    this->computeUnitsCompute_shm = UNIX_shm();
    this->computeUnitsDownload_shm = UNIX_shm();
    this->updateCnt_shm = UNIX_shm();
    this->computeCnt_shm = UNIX_shm();
    this->downloadCnt_shm = UNIX_shm();

    this->server_msq = UNIX_msg();
    this->client_msq = UNIX_msg();

    this->mValues = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;
    this->avCount = nullptr;
    this->avSet = nullptr;

    this->computeUnitsUpdate = nullptr;
    this->computeUnitsCompute = nullptr;
    this->computeUnitsDownload = nullptr;
    this->updateCnt = nullptr;
    this->computeCnt = nullptr;
    this->downloadCnt = nullptr;

    this->vValues_agent = nullptr;
    this->vSets_agent = nullptr;

    this->initPipeline(threadNum);
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::connect()
{
    int ret = 0;

    if (ret != -1) ret = this->mValues_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (MVALUES_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->initVSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (INITVSET_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->filteredV_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (FILTEREDV_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->timestamp_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (TIMESTAMP_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->avSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (AVSET_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->avCount_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (AVCOUNT_SHM << SHM_OFFSET)));

    if (ret != -1)
        ret = this->computeUnitsDownload_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (DOWNLOAD_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->computeUnitsUpdate_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (UPDATE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->computeUnitsCompute_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (COMPUTE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->downloadCnt_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (DOWNLOAD_CNT_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->updateCnt_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (UPDATE_CNT_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->computeCnt_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (COMPUTE_CNT_SHM << SHM_OFFSET)));


    if (ret != -1)
        ret = this->server_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (SRV_MSG_TYPE << MSG_TYPE_OFFSET)));
    if (ret != -1)
        ret = this->client_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (CLI_MSG_TYPE << MSG_TYPE_OFFSET)));

    if (ret != -1)
    {
        this->mValues_shm.attach(0666);
        this->initVSet_shm.attach(0666);
        this->filteredV_shm.attach(0666);
        this->timestamp_shm.attach(0666);
        this->avSet_shm.attach(0666);
        this->avCount_shm.attach(0666);

        this->computeUnitsCompute_shm.attach(0666);
        this->computeUnitsDownload_shm.attach(0666);
        this->computeUnitsUpdate_shm.attach(0666);
        this->computeCnt_shm.attach(0666);
        this->downloadCnt_shm.attach(0666);
        this->updateCnt_shm.attach(0666);

        this->mValues = (MessageValueType *) this->mValues_shm.shmaddr;
        this->initVSet = (int *) this->initVSet_shm.shmaddr;
        this->filteredV = (bool *) this->filteredV_shm.shmaddr;
        this->timestamp = (int *) this->timestamp_shm.shmaddr;
        this->avSet = (int *) this->avSet_shm.shmaddr;
        this->avCount = (int *) this->avCount_shm.shmaddr;

        this->computeUnitsUpdate = (ComputeUnit<VertexValueType> *) this->computeUnitsUpdate_shm.shmaddr;
        this->computeUnitsDownload = (ComputeUnit<VertexValueType> *) this->computeUnitsDownload_shm.shmaddr;
        this->computeUnitsCompute = (ComputeUnit<VertexValueType> *) this->computeUnitsCompute_shm.shmaddr;
        this->updateCnt = (int *) this->updateCnt_shm.shmaddr;
        this->downloadCnt = (int *) this->downloadCnt_shm.shmaddr;
        this->computeCnt = (int *) this->computeCnt_shm.shmaddr;
    }

    return ret;
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::transfer(VertexValueType *vValues, Vertex *vSet)
{
    if (vValues != nullptr) this->vValues_agent = vValues;
    if (vSet != nullptr) this->vSets_agent = vSet;
}

template<typename VertexValueType, typename MessageValueType>
int
UtilClient<VertexValueType, MessageValueType>::update(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits)
{
    if (computeUnitCount <= 0 || computeUnits == nullptr) return -1;
//    if (this->nodeNo == 0)
//        std::cout << "into update start " << std::endl;
//    if (this->nodeNo == 0)
//        std::cout << "computeUnitCount: " << computeUnitCount << std::endl;
    memcpy(this->computeUnitsUpdate, computeUnits, computeUnitCount * sizeof(ComputeUnit<VertexValueType>));
    memcpy(this->updateCnt, &computeUnitCount, sizeof(int));
//    if (this->nodeNo == 0)
//        std::cout << "this->updateCnt: " << *(this->updateCnt) << std::endl;
    return 0;
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::download()
{
//    if (this->nodeNo == 0)
//        std::cout << "download:" << std::endl;
    auto computeUnits = this->computeUnitsDownload;
    int count = *(this->downloadCnt);
//
//    if (this->nodeNo == 0)
//        std::cout << "download-cnt: " << count << std::endl;
//
//    if (this->nodeNo == 0)
//        std::cout << "address: " << &computeUnits[0] << std::endl;

    std::set<int> activeVertices;
    for (int i = 0; i < count; i++)
    {
        int destVId = computeUnits[i].destVertex.vertexID;
        int srcVId = computeUnits[i].srcVertex.vertexID;
        int indexOfInit = computeUnits[i].indexOfInitV;

        this->vSets_agent[destVId].isActive |= computeUnits[i].destVertex.isActive;
        this->vSets_agent[srcVId].isActive |= computeUnits[i].srcVertex.isActive;

        if (this->vSets_agent[destVId].isActive) activeVertices.emplace(destVId);
        if (this->vSets_agent[srcVId].isActive) activeVertices.emplace(srcVId);

        if (this->vValues_agent[srcVId * numOfInitV + indexOfInit] > computeUnits[i].srcValue)
            this->vValues_agent[srcVId * numOfInitV + indexOfInit] = computeUnits[i].srcValue;

        if (this->vValues_agent[destVId * numOfInitV + indexOfInit] > computeUnits[i].destValue)
            this->vValues_agent[destVId * numOfInitV + indexOfInit] = computeUnits[i].destValue;
    }



//    if (this->nodeNo == 0)
//        std::cout << "count:" << activeVertices.size() << std::endl;
//    if (this->nodeNo == 0){
//        for(auto id : activeVertices)
//        {
//            std::cout << "id:" << id << std::endl;
//        }
//
//    }
    return 0;
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::rotate()
{
    auto ptr = this->computeUnitsDownload;
    auto cntPtr = this->downloadCnt;

    this->computeUnitsDownload = this->computeUnitsCompute;
    this->computeUnitsCompute = this->computeUnitsUpdate;
    this->computeUnitsUpdate = ptr;

    this->downloadCnt = this->computeCnt;
    this->computeCnt = this->updateCnt;
    this->updateCnt = cntPtr;
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::disconnect()
{
    this->mValues_shm.detach();
    this->initVSet_shm.detach();
    this->filteredV_shm.detach();
    this->timestamp_shm.detach();
    this->avSet_shm.detach();
    this->avCount_shm.detach();

    this->computeUnitsCompute_shm.detach();
    this->computeUnitsDownload_shm.detach();
    this->computeUnitsUpdate_shm.detach();
    this->computeCnt_shm.detach();
    this->updateCnt_shm.detach();
    this->computeCnt_shm.detach();

    this->mValues = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;
    this->avSet = nullptr;
    this->avCount = nullptr;

    this->computeUnitsUpdate = nullptr;
    this->computeUnitsDownload = nullptr;
    this->computeUnitsCompute = nullptr;
    this->computeCnt = nullptr;
    this->updateCnt = nullptr;
    this->downloadCnt = nullptr;
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::shutdown()
{
    this->client_msq.send("exit", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    this->disconnect();
    this->stopPipeline();
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::graphInit()
{
    char tmp[256];

    this->client_msq.send("init", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    this->server_msq.recv(tmp, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::requestMSGApply()
{
    char tmp[256];
    this->client_msq.send("execute_msg_apply", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    while (this->server_msq.recv(tmp, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256))
    {
        if (std::string("finished msg apply") == tmp)
            return;

        if (errno == EINTR) continue;

        perror("msg apply");
        break;
    }
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::requestMSGMerge()
{
    char tmp[256];

    this->client_msq.send("execute_msg_merge", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    while (this->server_msq.recv(tmp, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256))
    {
        if (std::string("finished msg merge") == tmp)
            return;

        if (errno == EINTR) continue;

        perror("msg merge");
        break;
    }
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::initPipeline(int threadNum)
{
    if (this->threadPoolPtr == nullptr)
    {
        this->threadPoolPtr = std::make_shared<ThreadPool>(threadNum);
        this->threadPoolPtr->start();
    }
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::startPipeline(ComputeUnitPackage<VertexValueType> *computePackages,
                                                                  int packagesCnt)
{
    std::cout << "start pipeline" << std::endl;
    int sendCnt = 0;
    char serverMsg[256];

    //init message
    this->client_msq.send("IterationInit", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);

    //init the compute and download buffer
    *(this->computeCnt) = 0;
    *(this->downloadCnt) = 0;

    if (packagesCnt <= 0)
    {
        this->client_msq.send("End", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
        return;
    }

    //commit the update task
    if (sendCnt < packagesCnt)
    {
        auto computeUnits = computePackages[sendCnt].getUnitPtr();
        int count = computePackages[sendCnt].getCount();
        if (this->nodeNo == 0)
            std::cout << "count: " << count << std::endl;
        this->threadPoolPtr->commitTask(
                std::bind(&UtilClient<VertexValueType, MessageValueType>::update, this, count, computeUnits));
        sendCnt++;

        if (this->nodeNo == 0)
            std::cout << "update: " << *(this->updateCnt) << std::endl;
        while (this->threadPoolPtr->taskCount() != 0);
        if (this->nodeNo == 0)
            std::cout << "ExchangeF" << std::endl;
        this->client_msq.send("ExchangeF", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    }

    while (this->server_msq.recv(serverMsg, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256))
    {
        if (errno == EINTR) continue;

        if (!strcmp("RotateF", serverMsg))
        {
            if (this->nodeNo == 0)
                std::cout << "RotateF" << std::endl;
            rotate();

            // commit download task
            this->threadPoolPtr->commitTask(
                    std::bind(&UtilClient<VertexValueType, MessageValueType>::download, this));

            if (sendCnt == packagesCnt)
            {
                int a = 0;
                memcpy(this->updateCnt, &a, sizeof(int));
                continue;
            }

            // commit update task
            auto computeUnits = computePackages[sendCnt].getUnitPtr();
            int count = computePackages[sendCnt].getCount();
            this->threadPoolPtr->commitTask(
                    std::bind(&UtilClient<VertexValueType, MessageValueType>::update, this, count, computeUnits));

            sendCnt++;

            if (this->nodeNo == 0)
                std::cout << "RotateF" << std::endl;

        } else if (!strcmp("ComputeF", serverMsg))
        {
            if (this->nodeNo == 0)
                std::cout << "ComputeF" << std::endl;
            while (this->threadPoolPtr->taskCount() != 0);
            this->client_msq.send("ExchangeF", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);

            if (this->nodeNo == 0)
                std::cout << "ComputeF" << std::endl;
        } else if (!strcmp("ComputeAF", serverMsg))
        {
            if (this->nodeNo == 0)
                std::cout << "ComputeAF" << std::endl;
            while (this->threadPoolPtr->taskCount() != 0);
            break;
        }
    }
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::stopPipeline()
{
    this->threadPoolPtr->stop();
}