//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "UtilClient.h"

UtilClient::UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo)
{
    this->nodeNo = nodeNo;

    this->numOfInitV = numOfInitV;
    this->vCount = vCount;
    this->eCount = eCount;

    this->vValues_shm = UNIX_shm();
    this->eSrcSet_shm = UNIX_shm();
    this->eDstSet_shm = UNIX_shm();
    this->eWeightSet_shm = UNIX_shm();
    this->AVCheckSet_shm = UNIX_shm();
    this->initVSet_shm = UNIX_shm();

    this->server_msq = UNIX_msg();
    this->client_msq = UNIX_msg();
}

int UtilClient::connect()
{
    int ret = 0;

    if(ret != -1) ret = this->vValues_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->eSrcSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (ESRCSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->eDstSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (EDSTSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->eWeightSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (EWEIGHTSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->AVCheckSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (AVCHECKSET_SHM << SHM_OFFSET)));
    if(ret != -1) ret = this->initVSet_shm.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (INITVSET_SHM << SHM_OFFSET)));

    if(ret != -1) ret = this->server_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (SRV_MSG_TYPE << MSG_TYPE_OFFSET)));
    if(ret != -1) ret = this->client_msq.fetch(((this->nodeNo << NODE_NUM_OFFSET) | (CLI_MSG_TYPE << MSG_TYPE_OFFSET)));

    if(ret != -1)
    {
        this->vValues_shm.attach(0666);
        this->eSrcSet_shm.attach(0666);
        this->eDstSet_shm.attach(0666);
        this->eWeightSet_shm.attach(0666);
        this->AVCheckSet_shm.attach(0666);
        this->initVSet_shm.attach(0666);

        this->vValues = (double *)this->vValues_shm.shmaddr;
        this->eSrcSet = (int *)this->eSrcSet_shm.shmaddr;
        this->eDstSet = (int *)this->eDstSet_shm.shmaddr;
        this->eWeightSet = (double *)this->eWeightSet_shm.shmaddr;
        this->AVCheckSet = (bool *)this->AVCheckSet_shm.shmaddr;
        this->initVSet = (int *)this->initVSet_shm.shmaddr;
    }

    return ret;
}

int UtilClient::transfer(double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet, int *initVSet)
{
    if(this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
        memcpy(this->vValues, vValues, this->vCount * this->numOfInitV * sizeof(double));
        memcpy(this->eSrcSet, eSrcSet, this->eCount * sizeof(int));
        memcpy(this->eDstSet, eDstSet, this->eCount * sizeof(int));
        memcpy(this->eWeightSet, eWeightSet, this->eCount * sizeof(double));
        memcpy(this->AVCheckSet, AVCheckSet, this->vCount * sizeof(bool));
        memcpy(this->initVSet, initVSet, this->numOfInitV * sizeof(int));
        return 0;
    }
    else return -1;
}

int UtilClient::update(double *vValues, bool *AVCheckSet)
{
    if(this->vCount > 0 && this->eCount > 0 && this->numOfInitV > 0)
    {
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
    this->eSrcSet_shm.detach();
    this->eDstSet_shm.detach();
    this->eWeightSet_shm.detach();
    this->AVCheckSet_shm.detach();
    this->initVSet_shm.detach();
}

void UtilClient::shutdown()
{
    this->client_msq.send("exit", (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256);
    this->disconnect();
}
