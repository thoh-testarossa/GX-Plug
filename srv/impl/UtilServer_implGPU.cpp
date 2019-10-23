//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "../UtilServer.cpp"

#include "../../algo/BellmanFord/BellmanFordGPU.cpp"
#include "../../algo/ConnectedComponent/ConnectedComponentGPU.cpp"
#include "../../algo/PageRank/PageRankGPU.cpp"

template class UtilServer<BellmanFordGPU<double, double>, double, double>;
template class UtilServer<ConnectedComponentGPU<int, int>, int, int>;
template class UtilServer<PageRankGPU<std::pair<double, double>, PRA_MSG>, std::pair<double, double>, PRA_MSG>;
