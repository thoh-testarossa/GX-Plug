//
// Created by Thoh Testarossa on 2019-05-25.
//

#include "../GraphUtil.cpp"
#include "../../algo/LabelPropagation/LabelPropagation.h"


template class GraphUtil<double, double>;
template class GraphUtil<int, int>;
template class GraphUtil<std::pair<double, double>, std::pair<int, double>>;
template class GraphUtil<LPA_Value, std::pair<int, int>>;