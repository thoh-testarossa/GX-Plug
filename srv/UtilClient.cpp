//
// Created by Thoh Testarossa on 2019-04-05.
//

#include <cstring>
#include <iostream>
#include <memory>
#include "UtilClient.h"

template<typename VertexValueType, typename MessageValueType>
UtilClient<VertexValueType, MessageValueType>::UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo,
                                                          int maxComputeUnitsCnt)
{
    this->nodeNo = nodeNo;
    this->vCount = vCount;
    this->eCount = eCount;
    this->numOfInitV = numOfInitV;
    this->maxComputeUnitsCnt = maxComputeUnitsCnt;

    this->initVSet_shm = UNIX_shm();
    this->filteredV_shm = UNIX_shm();
    this->timestamp_shm = UNIX_shm();
    this->avCount_shm = UNIX_shm();
    this->avSet_shm = UNIX_shm();

    this->vValues_shm = UNIX_shm();
    this->mValues_shm = UNIX_shm();
    this->vSet_shm = UNIX_shm();
    this->eSet_shm = UNIX_shm();

    this->computeUnitsUpdate_shm = UNIX_shm();
    this->computeUnitsCompute_shm = UNIX_shm();
    this->computeUnitsDownload_shm = UNIX_shm();
    this->updateCnt_shm = UNIX_shm();
    this->computeCnt_shm = UNIX_shm();
    this->downloadCnt_shm = UNIX_shm();

    this->server_msq = UNIX_msg();
    this->client_msq = UNIX_msg();

    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;
    this->avCount = nullptr;
    this->avSet = nullptr;

    this->vValues = nullptr;
    this->mValues = nullptr;
    this->vSet = nullptr;
    this->eSet = nullptr;

    this->computeUnitsUpdate = nullptr;
    this->computeUnitsCompute = nullptr;
    this->computeUnitsDownload = nullptr;
    this->updateCnt = nullptr;
    this->computeCnt = nullptr;
    this->downloadCnt = nullptr;
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::connect()
{
    int ret = 0;

    if (ret != -1) ret = this->vValues_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->mValues_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (MVALUES_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->vSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VSET_SHM << SHM_OFFSET)));
    if (ret != -1) ret = this->eSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_SHM << SHM_OFFSET)));

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
        this->vValues_shm.attach(0666);
        this->mValues_shm.attach(0666);
        this->vSet_shm.attach(0666);
        this->eSet_shm.attach(0666);

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

        this->vValues = (VertexValueType *) this->vValues_shm.shmaddr;
        this->mValues = (MessageValueType *) this->mValues_shm.shmaddr;
        this->vSet = (Vertex *) this->vSet_shm.shmaddr;
        this->eSet = (Edge *) this->eSet_shm.shmaddr;

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
int UtilClient<VertexValueType, MessageValueType>::transfer(VertexValueType *vValues, Vertex *vSet, Edge *eSet,
                                                            int *initVSet, bool *filteredV, int *timestamp)
{
    if (this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if (this->vValues == nullptr) return -1;
        if (this->vSet == nullptr) return -1;
        if (this->eSet == nullptr) return -1;
        if (this->initVSet == nullptr) return -1;
        if (this->filteredV == nullptr) return -1;
        if (this->timestamp == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(VertexValueType));
        memcpy(this->vSet, vSet, this->vCount * sizeof(Vertex));
        memcpy(this->eSet, eSet, this->eCount * sizeof(Edge));
        memcpy(this->initVSet, initVSet, this->numOfInitV * sizeof(int));
        memcpy(this->filteredV, filteredV, this->vCount * sizeof(bool));
        memcpy(this->timestamp, timestamp, this->vCount * sizeof(int));

        this->graphInit();

        return 0;
    } else return -1;
}

template<typename VertexValueType, typename MessageValueType>
int
UtilClient<VertexValueType, MessageValueType>::update(VertexValueType *vValues, Vertex *vSet)
{
    if (this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if (this->vValues == nullptr) return -1;
        if (this->vSet == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(VertexValueType));
        memcpy(this->vSet, vSet, this->vCount * sizeof(Vertex));

        return 0;
    } else return -1;
}

template<typename VertexValueType, typename MessageValueType>
int
UtilClient<VertexValueType, MessageValueType>::pipeUpdate(int computeUnitCount,
                                                          ComputeUnit<VertexValueType> *computeUnits)
{
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
    this->vValues_shm.detach();
    this->mValues_shm.detach();
    this->vSet_shm.detach();
    this->eSet_shm.detach();

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

    this->vValues = nullptr;
    this->mValues = nullptr;
    this->vSet = nullptr;
    this->eSet = nullptr;

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
std::vector<ComputeUnitPackage<VertexValueType>> UtilClient<VertexValueType, MessageValueType>::computeUnitsGen()
{
    int computeCnt = 0;
    std::vector<ComputeUnitPackage<VertexValueType>> computePackages;
    ComputeUnit<VertexValueType> *computeUnits = nullptr;

    for (int i = 0; i < this->eCount; i++)
    {
        int destVId = this->eSet[i].dst;
        int srcVId = this->eSet[i].src;

        if (!this->vSet[srcVId].isActive) continue;
        if (this->numOfInitV <= 0) this->numOfInitV = 1;
        for (int j = 0; j < this->numOfInitV; j++)
        {
            if (computeCnt == 0) computeUnits = new ComputeUnit<VertexValueType>[this->maxComputeUnitsCnt];
            computeUnits[computeCnt].destVertex = this->vSet[destVId];
            computeUnits[computeCnt].destValue = this->vValues[destVId * this->numOfInitV + j];
            computeUnits[computeCnt].srcVertex = this->vSet[srcVId];
            computeUnits[computeCnt].srcValue = this->vValues[srcVId * this->numOfInitV + j];
            computeUnits[computeCnt].edgeWeight = this->eSet[i].weight;
            computeUnits[computeCnt].indexOfInitV = j;
            computeCnt++;
        }

        if (computeCnt == this->maxComputeUnitsCnt || i == this->eCount - 1)
        {
            computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
            computeCnt = 0;
        }
    }

    if (computeCnt != 0)
    {
        computePackages.emplace_back(ComputeUnitPackage<VertexValueType>(computeUnits, computeCnt));
    }

    return computePackages;
}


template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::startPipeline()
{
    std::cout << "start pipeline" << std::endl;

    int sendCnt = 0;
    int packagesCnt = 0;
    char serverMsg[256];
    std::vector<ComputeUnitPackage<VertexValueType>> computePackages;

    //package compute unit
    computePackages = this->computeUnitsGen();
    packagesCnt = computePackages.size();

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

        this->pipeUpdate(count, computeUnits);
        sendCnt++;

        this->client_msq.send("ExchangeF", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    }

    //iteration
    while (this->server_msq.recv(serverMsg, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256))
    {
        if (errno == EINTR) continue;

        if (!strcmp("RotateF", serverMsg))
        {
//            if (this->nodeNo == 0)
//                std::cout << "RotateF" << std::endl;
            rotate();

            if (sendCnt == packagesCnt)
            {
                *(this->updateCnt) = 0;
                this->client_msq.send("ExchangeAF", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
            } else
            {
                auto computeUnits = computePackages[sendCnt].getUnitPtr();
                int count = computePackages[sendCnt].getCount();

                this->pipeUpdate(count, computeUnits);
                sendCnt++;
                this->client_msq.send("ExchangeF", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
            }

        } else if (!strcmp("ComputeAF", serverMsg))
        {
            break;
        }
    }

    for (auto &computePackage : computePackages)
    {
        free(computePackage.getUnitPtr());
    }
}