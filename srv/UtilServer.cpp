//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "UtilServer.h"
#include <string>

template<typename T>
UtilServer<T>::UtilServer(int vCount, int eCount, int numOfInitV, int nodeNo)
{
    this->nodeNo = nodeNo;

    this->vCount = vCount;
    this->eCount = eCount;
    this->numOfInitV = numOfInitV;

    this->isLegal = TIsExtended<T, GraphUtil>::Result &&
                    vCount > 0 &&
                    eCount > 0 &&
                    numOfInitV > 0 &&
                    nodeNo >= 0;

    this->vValues = nullptr;
    this->eSrcSet = nullptr;
    this->eDstSet = nullptr;
    this->eWeightSet = nullptr;
    this->AVCheckSet = nullptr;
    this->initVSet = nullptr;

    if(this->isLegal)
    {
        this->numOfInitV = numOfInitV;
        this->vCount = vCount;
        this->eCount = eCount;

        this->executor = T();
        this->executor.Deploy(vCount, numOfInitV);

        this->vValues_shm = UNIX_shm();
        this->eSrcSet_shm = UNIX_shm();
        this->eDstSet_shm = UNIX_shm();
        this->eWeightSet_shm = UNIX_shm();
        this->AVCheckSet_shm = UNIX_shm();
        this->initVSet_shm = UNIX_shm();

        this->server_msq = UNIX_msg();
        this->client_msq = UNIX_msg();

        this->vValues_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (VVALUES_SHM << SHM_OFFSET)),
                this->vCount * this->numOfInitV * sizeof(double),
                0666);
        this->eSrcSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (ESRCSET_SHM << SHM_OFFSET)),
                this->eCount * sizeof(int),
                0666);
        this->eDstSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (EDSTSET_SHM << SHM_OFFSET)),
                this->eCount * sizeof(int),
                0666);
        this->eWeightSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (EWEIGHTSET_SHM << SHM_OFFSET)),
                this->eCount * sizeof(double),
                0666);
        this->AVCheckSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (AVCHECKSET_SHM << SHM_OFFSET)),
                this->vCount * sizeof(bool),
                0666);
        this->initVSet_shm.create(((this->nodeNo << NODE_NUM_OFFSET) | (INITVSET_SHM << SHM_OFFSET)),
                this->numOfInitV * sizeof(int),
                0666);

        this->server_msq.create(((this->nodeNo << NODE_NUM_OFFSET) | (SRV_MSG_TYPE << MSG_TYPE_OFFSET)),
                0666);
        this->client_msq.create(((this->nodeNo << NODE_NUM_OFFSET) | (CLI_MSG_TYPE << MSG_TYPE_OFFSET)),
                0666);

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
}

template<typename T>
UtilServer<T>::~UtilServer()
{
    if(this->isLegal)
    {
        this->executor.Free();

        this->vValues = nullptr;
        this->eSrcSet = nullptr;
        this->eDstSet = nullptr;
        this->eWeightSet = nullptr;
        this->AVCheckSet = nullptr;
        this->initVSet = nullptr;

        this->vValues_shm.control(IPC_RMID);
        this->eSrcSet_shm.control(IPC_RMID);
        this->eDstSet_shm.control(IPC_RMID);
        this->eWeightSet_shm.control(IPC_RMID);
        this->AVCheckSet_shm.control(IPC_RMID);
        this->initVSet_shm.control(IPC_RMID);

        this->server_msq.control(IPC_RMID);
        this->client_msq.control(IPC_RMID);
    }
}

template<typename T>
void UtilServer<T>::run()
{
    if(!this->isLegal) return;

    double *mValues = new double [this->vCount * this->numOfInitV];
    char msgp[256];
    std::string cmd = std::string("");

    while(this->client_msq.recv(msgp, (CLI_MSG_TYPE << MSG_TYPE_OFFSET), 256) != -1)
    {
        cmd = msgp;
        if(std::string("execute") == cmd)
        {
            for (int i = 0; i < this->vCount * this->numOfInitV; i++) mValues[i] = INVALID_MASSAGE;

            this->executor.MSGGenMerge_array(this->vCount, this->eCount, this->numOfInitV, this->initVSet, this->vValues, this->eSrcSet, this->eDstSet, this->eWeightSet, mValues, this->AVCheckSet);
            this->executor.MSGApply_array(this->vCount, this->numOfInitV, this->initVSet, this->AVCheckSet, this->vValues, mValues);

            this->server_msq.send("finished", (SRV_MSG_TYPE << MSG_TYPE_OFFSET), 256);
        }

        else if(std::string("exit") == cmd)
            break;
        else break;
    }
}

