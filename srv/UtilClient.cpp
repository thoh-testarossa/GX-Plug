//
// Created by Thoh Testarossa on 2019-04-05.
//

#include <cstring>
#include <iostream>
#include "UtilClient.h"

template<typename VertexValueType, typename MessageValueType>
UtilClient<VertexValueType, MessageValueType>::UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo,
                                                          int pipeVCount, int pipeECount)
{
    this->nodeNo = nodeNo;

    this->numOfInitV = numOfInitV;
    this->totalVCount = vCount;
    this->totalECount = eCount;
    this->pipeECount = pipeECount;
    this->pipeVCount = pipeVCount;

    this->mValues_shm = UNIX_shm();
    this->initVSet_shm = UNIX_shm();
    this->filteredV_shm = UNIX_shm();
    this->timestamp_shm = UNIX_shm();
    this->avCount_shm = UNIX_shm();
    this->avSet_shm = UNIX_shm();

    this->vValuesUpdate_shm = UNIX_shm();
    this->vValuesDownload_shm = UNIX_shm();
    this->vValuesCompute_shm = UNIX_shm();
    this->vSetUpdate_shm = UNIX_shm();
    this->vSetDownload_shm = UNIX_shm();
    this->vSetCompute_shm = UNIX_shm();
    this->eSetUpdate_shm = UNIX_shm();
    this->eSetDownload_shm = UNIX_shm();
    this->eSetCompute_shm = UNIX_shm();

    this->server_msq = UNIX_msg();
    this->client_msq = UNIX_msg();

    this->mValues = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;
    this->avCount = nullptr;
    this->avSet = nullptr;

    this->vValuesUpdate = nullptr;
    this->vValuesDownload = nullptr;
    this->vValuesCompute = nullptr;
    this->vSetUpdate = nullptr;
    this->vSetDownload = nullptr;
    this->vSetCompute = nullptr;
    this->eSetUpdate = nullptr;
    this->eSetDownload = nullptr;
    this->eSetCompute = nullptr;
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
        ret = this->vValuesUpdate_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_UPDATE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->vValuesDownload_shm.fetch(
                ((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_DOWNLOAD_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->vValuesCompute_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_COMPUTE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->vSetUpdate_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VSET_UPDATE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->vSetDownload_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VSET_DOWNLOAD_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->vSetCompute_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VSET_COMPUTE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->eSetUpdate_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_UPDATE_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->eSetDownload_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_DOWNLOAD_SHM << SHM_OFFSET)));
    if (ret != -1)
        ret = this->eSetCompute_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_COMPUTE_SHM << SHM_OFFSET)));
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

        this->vValuesUpdate_shm.attach(0666);
        this->vValuesDownload_shm.attach(0666);
        this->vValuesCompute_shm.attach(0666);
        this->vSetUpdate_shm.attach(0666);
        this->vSetDownload_shm.attach(0666);
        this->vSetCompute_shm.attach(0666);
        this->eSetUpdate_shm.attach(0666);
        this->eSetDownload_shm.attach(0666);
        this->eSetCompute_shm.attach(0666);

        this->mValues = (MessageValueType *) this->mValues_shm.shmaddr;
        this->initVSet = (int *) this->initVSet_shm.shmaddr;
        this->filteredV = (bool *) this->filteredV_shm.shmaddr;
        this->timestamp = (int *) this->timestamp_shm.shmaddr;
        this->avSet = (int *) this->avSet_shm.shmaddr;
        this->avCount = (int *) this->avCount_shm.shmaddr;

        this->vValuesUpdate = (VertexValueType *) this->vValuesUpdate_shm.shmaddr;
        this->vValuesDownload = (VertexValueType *) this->vValuesDownload_shm.shmaddr;
        this->vValuesCompute = (VertexValueType *) this->vValuesCompute_shm.shmaddr;
        this->vSetUpdate = (Vertex *) this->vSetUpdate_shm.shmaddr;
        this->vSetDownload = (Vertex *) this->vSetDownload_shm.shmaddr;
        this->vSetCompute = (Vertex *) this->vSetCompute_shm.shmaddr;
        this->eSetUpdate = (Edge *) this->eSetUpdate_shm.shmaddr;
        this->eSetDownload = (Edge *) this->eSetDownload_shm.shmaddr;
        this->eSetCompute = (Edge *) this->eSetCompute_shm.shmaddr;
    }

    return ret;
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::transfer(int *initVSet, bool *filteredV, int *timestamp)
{
    if (this->numOfInitV > 0)
    {
        if (this->initVSet == nullptr) return -1;
        if (this->filteredV == nullptr) return -1;
        if (this->timestamp == nullptr) return -1;

        memcpy(this->initVSet, initVSet, this->numOfInitV * sizeof(int));
        memcpy(this->filteredV, filteredV, this->vCount * sizeof(bool));
        memcpy(this->timestamp, timestamp, this->vCount * sizeof(int));

        this->graphInit();

        return 0;
    } else return -1;
}

template<typename VertexValueType, typename MessageValueType>
int
UtilClient<VertexValueType, MessageValueType>::update(VertexValueType *vValues, Vertex *vSet, int *avSet, int avCount)
{
    if (this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if (this->vValues == nullptr) return -1;
        if (this->vSet == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(VertexValueType));
        memcpy(this->vSet, vSet, this->vCount * sizeof(Vertex));

        memcpy(this->avCount, &avCount, sizeof(int));

        if (avCount > 0 && avSet != nullptr)
        {
            memcpy(this->avSet, avSet, avCount * sizeof(int));
        }

        return 0;
    } else return -1;
}

template<typename VertexValueType, typename MessageValueType>
int UtilClient<VertexValueType, MessageValueType>::update(VertexValueType *vValues, int *avSet, int avCount)
{
    if (this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if (this->vValues == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(VertexValueType));

        memcpy(this->avCount, &avCount, sizeof(int));

        if (avCount > 0 && avSet != nullptr)
        {
            memcpy(this->avSet, avSet, avCount * sizeof(int));
        }

        return 0;
    } else return -1;
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

    this->vValuesUpdate_shm.detach();
    this->vValuesDownload_shm.detach();
    this->vValuesCompute_shm.detach();
    this->vSetUpdate_shm.detach();
    this->vSetDownload_shm.detach();
    this->vSetCompute_shm.detach();
    this->eSetUpdate_shm.detach();
    this->eSetDownload_shm.detach();
    this->eSetCompute_shm.detach();

    this->mValues = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;
    this->avSet = nullptr;
    this->avCount = nullptr;

    this->vValuesUpdate = nullptr;
    this->vValuesDownload = nullptr;
    this->vValuesCompute = nullptr;
    this->vSetUpdate = nullptr;
    this->vSetDownload = nullptr;
    this->vSetCompute = nullptr;
    this->eSetUpdate = nullptr;
    this->eSetDownload = nullptr;
    this->eSetCompute = nullptr;
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
        this->threadPoolPtr = new ThreadPool(threadNum);
        this->threadPoolPtr->start();
    }
}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::startPipeline(VertexValueType *vValues, Vertex *vSet, Edge *eSet)
{

}

template<typename VertexValueType, typename MessageValueType>
void UtilClient<VertexValueType, MessageValueType>::stopPipeline()
{
    this->threadPoolPtr->stop();
}