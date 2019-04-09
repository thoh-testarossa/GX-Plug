//
// Created by Thoh Testarossa on 2019-04-05.
//

#pragma once

#ifndef GRAPH_ALGO_UTILSERVER_H
#define GRAPH_ALGO_UTILSERVER_H

#include "../core/GraphUtil.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../include/UNIX_marco.h"

template<typename T>
class UtilServer
{
public:
    UtilServer(int vCount, int eCount, int numOfInitV, int nodeNo = 0);
    ~UtilServer();

    void run();

    int nodeNo;
    T executor;
    bool isLegal;

    int vCount;
    int eCount;
    int numOfInitV;

    int *initVSet;
    double *vValues;
    int *eSrcSet;
    int *eDstSet;
    double *eWeightSet;
    bool *AVCheckSet;

private:
    UNIX_shm initVSet_shm;
    UNIX_shm vValues_shm;
    UNIX_shm eSrcSet_shm;
    UNIX_shm eDstSet_shm;
    UNIX_shm eWeightSet_shm;
    UNIX_shm AVCheckSet_shm;

    UNIX_msg server_msq;
    UNIX_msg client_msq;
};

#endif //GRAPH_ALGO_UTILSERVER_H
