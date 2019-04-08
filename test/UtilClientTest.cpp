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

#include <future>

void testFut(UtilClient *uc, double *vValues, bool *AVCheckSet)
{
    uc->connect();
    uc->update(vValues, AVCheckSet);
    uc->request();
    uc->disconnect();
}

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
        clientVec.push_back(UtilClient(vCount, ((i + 1) * eCount) / nodeCount - (i * eCount) / nodeCount, numOfInitV, i));
    int chk = 0;
    for(int i = 0; i < nodeCount && chk != -1; i++)
    {
        chk = clientVec.at(i).connect();
        if (chk == -1)
        {
            std::cout << "Cannot establish the connection with server correctly" << std::endl;
            return 2;
        }

        chk = clientVec.at(i).transfer(vValues, &eSrcSet[(i * eCount) / nodeCount], &eDstSet[(i * eCount) / nodeCount], &eWeightSet[(i * eCount) / nodeCount], AVCheckSet, initVSet);

        if(chk == -1)
        {
            std::cout << "Parameter illegal" << std::endl;
            return 3;
        }

        clientVec.at(i).disconnect();
    }

    bool isActive = false;
    for(int i = 0; i < vCount; i++) isActive |= AVCheckSet[i];

    bool *ret_AVCheckSet = new bool [vCount];
    int iterCount = 0;

    while(isActive)
    {
        //Test
        std::cout << "Processing at iter " << ++iterCount << std::endl;
        //Test end

        for(int i = 0; i < vCount; i++) ret_AVCheckSet[i] = false;

        auto futList = new std::future<void> [nodeCount];
        for(int i = 0; i < nodeCount; i++)
        {
            std::future<void> tmpFut = std::async(testFut, &clientVec.at(i), vValues, AVCheckSet);
            futList[i] = std::move(tmpFut);
        }

        for(int i = 0; i < nodeCount; i++)
            futList[i].get();

        //Retrieve data
        for(int i = 0; i < nodeCount; i++)
        {
            clientVec.at(i).connect();

            //Collect data
            for(int j = 0; j < vCount * numOfInitV; j++)
            {
                if (clientVec.at(i).vValues[j] < vValues[j])
                    vValues[j] = clientVec.at(i).vValues[j];
            }

            for(int j = 0; j < vCount; j++)
                ret_AVCheckSet[j] |= clientVec.at(i).AVCheckSet[j];

            clientVec.at(i).disconnect();
        }

        memcpy(AVCheckSet, ret_AVCheckSet, vCount);

        isActive = false;
        for(int i = 0; i < vCount; i++) isActive |= AVCheckSet[i];

        //Test
        for(int i = 0; i < vCount * numOfInitV; i++)
        {
            std::cout << vValues[i] << " ";
            if(i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
        }
        for(int i = 0; i < vCount; i++)
            std::cout << AVCheckSet[i];
        std::cout << std::endl;
        std::cout << isActive << std::endl;
        //Test end
    }

    for(int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();

    //result check
    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        std::cout << vValues[i] << " ";
        if(i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
    }
}