//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../algo/BellmanFord/BellmanFord.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilServer.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>

int main(int argc, char *argv[])
{
    if(argc != 5 && argc != 6)
    {
        std::cout << "Usage:" << std::endl
        << "./UtilServerTest_BellmanFord vCount eCount numOfInitV maxComputeUnits [nodeNo]" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int numOfInitV = atoi(argv[3]);
    int maxComputeUnits = atoi(argv[4]);
    int nodeNo = (argc == 5) ? 0 : atoi(argv[5]);

    auto testUtilServer = UtilServer<BellmanFord<double, double>, double, double>
            (vCount, eCount, numOfInitV, nodeNo, maxComputeUnits);

    if(!testUtilServer.isLegal)
    {
        std::cout << "mem allocation failed or parameters are illegal" << std::endl;
        return 2;
    }

    testUtilServer.run();
}