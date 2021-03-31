//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "../srv/UtilServer.h"
#include "../util/TIsExtended.hpp"
#include <string>
#include <iostream>
#include <chrono>

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
UtilServer<GraphUtilType, VertexValueType, MessageValueType>::UtilServer(int vCount, int eCount, int numOfInitV,
                                                                         int nodeNo, int maxComputeUnits)
{
    //Test
    std::cout << "Server init" << std::endl;
    //Test end

    this->nodeNo = nodeNo;

    this->vCount = vCount;
    this->eCount = eCount;
    this->numOfInitV = numOfInitV;
    this->maxComputeUnits = maxComputeUnits;

    this->isLegal = TIsExtended<GraphUtilType, GraphUtil<VertexValueType, MessageValueType>>::Result &&
                    vCount > 0 &&
                    eCount > 0 &&
                    numOfInitV > 0 &&
                    nodeNo >= 0 &&
                    maxComputeUnits >= 0;


    this->vValues = nullptr;
    this->mValues = nullptr;
    this->vSet = nullptr;
    this->eSet = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;

    this->computeUnitsUpdate = nullptr;
    this->computeUnitsCompute = nullptr;
    this->computeUnitsDownload = nullptr;
    this->updateCnt = nullptr;
    this->computeCnt = nullptr;
    this->downloadCnt = nullptr;

    this->initPipeline(2);

    if (this->isLegal)
    {
        int chk = 0;

        this->executor = GraphUtilType();
        this->executor.Init(vCount, eCount, numOfInitV, maxComputeUnits);
        this->executor.partitionId = nodeNo;

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
        this->init_msq = UNIX_msg();
        this->client_msq = UNIX_msg();

        if (chk != -1)
            chk = this->vValues_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_SHM << SHM_OFFSET)),
                                           this->executor.totalVValuesCount * sizeof(VertexValueType),
                                           0666);
        if (chk != -1)
            chk = this->mValues_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (MVALUES_SHM << SHM_OFFSET)),
                                           this->executor.totalMValuesCount * sizeof(MessageValueType),
                                           0666);
        if (chk != -1)
            chk = this->vSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (VSET_SHM << SHM_OFFSET)),
                                        this->vCount * sizeof(Vertex),
                                        0666);
        if (chk != -1)
            chk = this->eSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_SHM << SHM_OFFSET)),
                                        this->eCount * sizeof(Edge),
                                        0666);
        if (chk != -1)
            chk = this->initVSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (INITVSET_SHM << SHM_OFFSET)),
                                            this->numOfInitV * sizeof(int),
                                            0666);
        if (chk != -1)
            chk = this->filteredV_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (FILTEREDV_SHM << SHM_OFFSET)),
                                             this->vCount * sizeof(bool),
                                             0666);
        if (chk != -1)
            chk = this->timestamp_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (TIMESTAMP_SHM << SHM_OFFSET)),
                                             this->vCount * sizeof(int),
                                             0666);
        if (chk != -1)
            chk = this->avSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (AVSET_SHM << SHM_OFFSET)),
                                         this->vCount * sizeof(int),
                                         0666);
        if (chk != -1)
            chk = this->avCount_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (AVCOUNT_SHM << SHM_OFFSET)),
                                           sizeof(int),
                                           0666);

        if (chk != -1)
            chk = this->computeUnitsDownload_shm.create(
                    ((this->nodeNo << NODE_NUM_OFFSET) | (DOWNLOAD_SHM << SHM_OFFSET)),
                    this->maxComputeUnits * sizeof(ComputeUnit<VertexValueType>), 0666);
        if (chk != -1)
            chk = this->computeUnitsUpdate_shm.create(
                    ((this->nodeNo << NODE_NUM_OFFSET) | (UPDATE_SHM << SHM_OFFSET)),
                    this->maxComputeUnits * sizeof(ComputeUnit<VertexValueType>), 0666);
        if (chk != -1)
            chk = this->computeUnitsCompute_shm.create(
                    ((this->nodeNo << NODE_NUM_OFFSET) | (COMPUTE_SHM << SHM_OFFSET)),
                    this->maxComputeUnits * sizeof(ComputeUnit<VertexValueType>), 0666);
        if (chk != -1)
            chk = this->downloadCnt_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (DOWNLOAD_CNT_SHM << SHM_OFFSET)),
                                               sizeof(int), 0666);
        if (chk != -1)
            chk = this->updateCnt_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (UPDATE_CNT_SHM << SHM_OFFSET)),
                                             sizeof(int), 0666);
        if (chk != -1)
            chk = this->computeCnt_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (COMPUTE_CNT_SHM << SHM_OFFSET)),
                                              sizeof(int), 0666);

        if (chk != -1)
            chk = this->server_msq.create(((this->nodeNo << NODE_NUM_OFFSET) | (SRV_MSG_TYPE << MSG_TYPE_OFFSET)),
                                          0666);
        if (chk != -1)
            chk = this->init_msq.create(((this->nodeNo << NODE_NUM_OFFSET) | (INIT_MSG_TYPE << MSG_TYPE_OFFSET)),
                                        0666);
        if (chk != -1)
            chk = this->client_msq.create(((this->nodeNo << NODE_NUM_OFFSET) | (CLI_MSG_TYPE << MSG_TYPE_OFFSET)),
                                          0666);

        if (chk != -1)
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

            this->mValues = (MessageValueType *) this->mValues_shm.shmaddr;
            this->vValues = (VertexValueType *) this->vValues_shm.shmaddr;
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

            this->init_msq.send("initiated", (INIT_MSG_TYPE << MSG_TYPE_OFFSET), 256);

            //Test
            std::cout << "Init succeeded." << std::endl;
            //Test end

            this->executor.Deploy(vCount, eCount, numOfInitV);
        } else
        {
            this->isLegal = false;

            //Test
            std::cout << "Init failed with errno " << errno << std::endl;
            //Test end
        }
    }
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
UtilServer<GraphUtilType, VertexValueType, MessageValueType>::~UtilServer()
{
    this->executor.Free();

    this->vValues = nullptr;
    this->mValues = nullptr;
    this->vSet = nullptr;
    this->eSet = nullptr;
    this->initVSet = nullptr;
    this->filteredV = nullptr;
    this->timestamp = nullptr;

    this->computeUnitsUpdate = nullptr;
    this->computeUnitsDownload = nullptr;
    this->computeUnitsCompute = nullptr;
    this->computeCnt = nullptr;
    this->updateCnt = nullptr;
    this->downloadCnt = nullptr;

    this->mValues_shm.control(IPC_RMID);
    this->vValues_shm.control(IPC_RMID);
    this->vSet_shm.control(IPC_RMID);
    this->eSet_shm.control(IPC_RMID);
    this->initVSet_shm.control(IPC_RMID);
    this->filteredV_shm.control(IPC_RMID);
    this->timestamp_shm.control(IPC_RMID);
    this->avSet_shm.control(IPC_RMID);
    this->avCount_shm.control(IPC_RMID);

    this->computeUnitsCompute_shm.control(IPC_RMID);
    this->computeUnitsUpdate_shm.control(IPC_RMID);
    this->computeUnitsDownload_shm.control(IPC_RMID);
    this->computeCnt_shm.control(IPC_RMID);
    this->updateCnt_shm.control(IPC_RMID);
    this->downloadCnt_shm.control(IPC_RMID);

    this->server_msq.control(IPC_RMID);
    this->init_msq.control(IPC_RMID);
    this->client_msq.control(IPC_RMID);
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::rotate()
{
    auto ptr = this->computeUnitsDownload;
    auto cntPtr = this->downloadCnt;

//    std::cout << "update: " << *(this->updateCnt) << std::endl;
//    std::cout << "computeCnt: " << *(this->computeCnt) << std::endl;

    this->computeUnitsDownload = this->computeUnitsCompute;
    this->computeUnitsCompute = this->computeUnitsUpdate;
    this->computeUnitsUpdate = ptr;

    this->downloadCnt = this->computeCnt;
    this->computeCnt = this->updateCnt;
    this->updateCnt = cntPtr;

//    std::cout << "update: " << *(this->updateCnt) << std::endl;
//    std::cout << "computeCnt: " << *(this->computeCnt) << std::endl;
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::initPipeline(int threadNum)
{
    if (this->threadPoolPtr == nullptr)
    {
        this->threadPoolPtr = std::make_shared<ThreadPool>(threadNum);
        this->threadPoolPtr->start();
    }
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::pipeCompute()
{
//    std::cout << "pipeCompute" << std::endl;

    if (*(this->computeCnt) == 0) return;

    this->executor.MSGGenMerge_array(*(this->computeCnt), this->computeUnitsCompute, this->mValues);
    this->executor.MSGApply_array(*(this->computeCnt), this->computeUnitsCompute, this->mValues);
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::pipeDownload()
{
//    std::cout << "pipeDownload" << std::endl;
    if (*(this->downloadCnt) == 0) return;
    this->executor.download(this->vValues, this->vSet, *(this->downloadCnt), this->computeUnitsDownload);
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::stopPipeline()
{
    this->threadPoolPtr->stop();
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::run()
{
    if (!this->isLegal) return;

    char msgp[256];
    std::string cmd = std::string("");

    int iterCount = 0;

    int msgCount = 0;
    int avCount = 0;

    //test
    int total = 0;

    while (this->client_msq.recv(msgp, (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256) != -1)
    {
        //Test
        // std::cout << "Processing at iter " << iterCount << std::endl;
        //Test end

        cmd = msgp;
        if (std::string("init") == cmd)
        {
            auto start = std::chrono::system_clock::now();

            if (this->executor.optimize)
                this->graphInit();
            this->server_msq.send("finished", (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);

            auto end = std::chrono::system_clock::now();

            std::cout << "server init time : "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } else if (std::string("ExchangeF") == cmd || std::string("ExchangeAF") == cmd)
        {
//            std::cout << "RotateF" << std::endl;
            //sync
            while (this->threadPoolPtr->taskCount() != 0);

            rotate();

            if (std::string("ExchangeF") == cmd)
                this->server_msq.send("RotateF", (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);

            //commit download task
            this->threadPoolPtr->commitTask(
                    std::bind(&UtilServer<GraphUtilType, VertexValueType, MessageValueType>::pipeDownload, this));

            //commit compute task
            this->threadPoolPtr->commitTask(
                    std::bind(&UtilServer<GraphUtilType, VertexValueType, MessageValueType>::pipeCompute, this));

            if (std::string("ExchangeAF") == cmd)
            {
                while (this->threadPoolPtr->taskCount() != 0);
                this->server_msq.send("ComputeAF", (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);
                this->executor.IterationEnd(this->mValues);
                iterCount++;
            }

//            std::cout << "RotateF" << std::endl;
        } else if (std::string("IterationInit") == cmd)
        {
            this->executor.IterationInit(this->vCount, this->eCount, this->mValues);

            //init active vertex
            for (int i = 0; i < this->vCount; i++)
                this->vSet[i].isActive = false;

        } else if (std::string("exit") == cmd)
        {
            this->stopPipeline();
            break;
        }
    }

    //Test
    std::cout << "total time: " << total << std::endl;
    std::cout << "Shutdown properly" << std::endl;
    //Test end
}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
void UtilServer<GraphUtilType, VertexValueType, MessageValueType>::graphInit()
{

}

template<typename GraphUtilType, typename VertexValueType, typename MessageValueType>
bool UtilServer<GraphUtilType, VertexValueType, MessageValueType>::getEdgesFromAvSet()
{

}

