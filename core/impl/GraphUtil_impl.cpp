//
// Created by Thoh Testarossa on 2019-05-25.
//

#include "../GraphUtil.cpp"
#include "../../algo/LabelPropagation/LabelPropagation.h"
#include "../../algo/PageRank/PageRank.h"


template class GraphUtil<double, double>;
template class GraphUtil<int, int>;
template class GraphUtil<std::pair<double, double>, PRA_MSG>;
template class GraphUtil<LPA_Value, std::pair<int, int>>;