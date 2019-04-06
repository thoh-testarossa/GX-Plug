//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char *argv[])
{
    if(argc != 4 && argc != 5)
    {
        std::cout << "Usage:" << std::endl << "./UtilClientTest_BellmanFord vCount eCount numOfInitV [nodeCount]" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int numOfInitV = atoi(argv[3]);
    int nodeCount = (argc == 4) ? 1 : atoi(argv[4]);

    //Parameter check
    if(vCount <= 0 || eCount <= 0 || numOfInitV <= 0 || nodeCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    //Init the Graph
    bool *AVCheckSet = new bool [vCount];
    int *initVSet = new int [numOfInitV];
    double *vValues = new double [vCount * numOfInitV];

    int *eSrcSet = new int [eCount];
    int *eDstSet = new int [eCount];
    double *eWeightSet = new double [eCount];

    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    int tmp;
    Gin >> tmp;
    if(vCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }
    Gin >> tmp;
    if(eCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }

    for(int i = 0; i < vCount * numOfInitV; i++) vValues[i] = INT32_MAX  >> 1;
    for(int i = 0; i < eCount; i++) Gin >> eSrcSet[i] >> eDstSet[i] >> eWeightSet[i];
    //Easy init
    for(int i = 0; i < numOfInitV; i++) initVSet[i] = i;
    for(int i = 0; i < vCount; i++) AVCheckSet[i] = i < numOfInitV;
    for(int i = 0; i < numOfInitV; i++) vValues[i * numOfInitV + i] = 0;

    Gin.close();

    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient>();
    for(int i = 0; i < nodeCount; i++)
        clientVec.push_back(UtilClient(vCount, eCount, numOfInitV, i));
    int chk = 0;
    for(int i = 0; i < nodeCount && chk != -1; i++)
    {
        chk = clientVec.at(i).connect();
        if (chk == -1)
        {
            std::cout << "Cannot establish the connection with server correctly" << std::endl;
            return 2;
        }

        chk = clientVec.at(i).transfer(vValues, eSrcSet, eDstSet, eWeightSet, AVCheckSet, initVSet);

        if(chk == -1)
        {
            std::cout << "Parameter illegal" << std::endl;
            return 3;
        }

        clientVec.at(i).disconnect();
    }

    bool isActive = false;
    for(int i = 0; i < vCount; i++) isActive |= AVCheckSet[i];

    while(isActive)
    {
        for(int i = 0; i < nodeCount; i++)
        {
            clientVec.at(i).connect();
            clientVec.at(i).update(vValues, AVCheckSet);
            clientVec.at(i).request();

            //Collect data
            for(int j = 0; j < vCount * numOfInitV; j++)
            {
                if (clientVec.at(i).vValues[j] < vValues[j])
                    vValues[j] = clientVec.at(i).vValues[j];
            }

            for(int j = 0; j < vCount; j++)
                AVCheckSet[j] = false | clientVec.at(i).AVCheckSet[j];

            clientVec.at(i).disconnect();
        }

        isActive = false;
        for(int i = 0; i < vCount; i++) isActive |= AVCheckSet[i];
    }

    for(int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();

    //result check
    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        std::cout << vValues[i] << " ";
        if(i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
    }
}