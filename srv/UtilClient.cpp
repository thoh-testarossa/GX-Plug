//
// Created by Thoh Testarossa on 2019-04-05.
//

#include <cstring>
#include "UtilClient.h"

UtilClient::UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo)
{
    this->nodeNo = nodeNo;

    this->numOfInitV = numOfInitV;
    this->vCount = vCount;
    this->eCount = eCount;

    this->vValues_shm = UNIX_shm();
    this->eSet_shm = UNIX_shm();
    this->AVCheckSet_shm = UNIX_shm();
    this->initVSet_shm = UNIX_shm();
    this->initVIndexSet_shm = UNIX_shm();

    this->server_msq = UNIX_msg();
    this->client_msq = UNIX_msg();

    this->vValues = nullptr;
    this->eSet = nullptr;
    this->AVCheckSet = nullptr;
    this->initVSet = nullptr;
    this->initVIndexSet = nullptr;
}

int UtilClient::connect()
{
    int ret = 0;

    if(ret != -1) ret = this->vValues_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->eSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->AVCheckSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (AVCHECKSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->initVSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (INITVSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->initVIndexSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (INITVINDEXSET_SHM << SHM_OFFSET)));

    if(ret != -1) ret = this->server_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (SRV_MSG_TYPE << MSG_TYPE_OFFSET)));
    if(ret != -1) ret = this->client_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (CLI_MSG_TYPE << MSG_TYPE_OFFSET)));

    if(ret != -1)
    {
        this->vValues_shm.attach(0666);
        this->eSet_shm.attach(0666);
        this->AVCheckSet_shm.attach(0666);
        this->initVSet_shm.attach(0666);
        this->initVIndexSet_shm.attach(0666);

        this->vValues = (double *)this->vValues_shm.shmaddr;
        this->eSet = (Edge *)this->eSet_shm.shmaddr;
        this->AVCheckSet = (bool *)this->AVCheckSet_shm.shmaddr;
        this->initVSet = (int *)this->initVSet_shm.shmaddr;
        this->initVIndexSet = (int *)this->initVIndexSet_shm.shmaddr;
    }

    return ret;
}

int UtilClient::transfer(double *vValues, Edge *eSet, bool *AVCheckSet, int *initVSet, int *initVIndexSet)
{
    if(this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if(this->vValues == nullptr) return -1;
        if(this->eSet == nullptr) return -1;
        if(this->AVCheckSet == nullptr) return -1;
        if(this->initVSet == nullptr) return -1;
        if(this->initVIndexSet == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(double));
        memcpy(this->eSet, eSet, this->eCount * sizeof(Edge));
        memcpy(this->AVCheckSet, AVCheckSet, this->vCount * sizeof(bool));
        memcpy(this->initVSet, initVSet, this->numOfInitV * sizeof(int));
        memcpy(this->initVIndexSet, initVIndexSet, this->vCount * sizeof(int));
        return 0;
    }
    else return -1;
}

int UtilClient::update(double *vValues, bool *AVCheckSet)
{
    if(this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        if(this->vValues == nullptr) return -1;
        if(this->AVCheckSet == nullptr) return -1;

        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(double));
        memcpy(this->AVCheckSet, AVCheckSet, this->vCount * sizeof(bool));
        return 0;
    }
    else return -1;
}

void UtilClient::request()
{
    char tmp[256];

    this->client_msq.send("execute", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    this->server_msq.recv(tmp, (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);
}

void UtilClient::disconnect()
{
    this->vValues_shm.detach();
    this->eSet_shm.detach();
    this->AVCheckSet_shm.detach();
    this->initVSet_shm.detach();
    this->initVIndexSet_shm.detach();

    this->vValues = nullptr;
    this->eSet = nullptr;
    this->AVCheckSet = nullptr;
    this->initVSet = nullptr;
    this->initVIndexSet = nullptr;
}

void UtilClient::shutdown()
{
    this->client_msq.send("exit", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    this->disconnect();
}
