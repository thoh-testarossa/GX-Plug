//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "../UtilServer.cpp"

#include "../../algo/BellmanFord/BellmanFord.cpp"
#include "../../algo/LabelPropagation/LabelPropagation.cpp"

template class UtilServer<BellmanFord<double, double>, double, double>;
template class UtilServer<LabelPropagation<LPA_Value, std::pair<int, int>>, LPA_Value, std::pair<int, int>>;