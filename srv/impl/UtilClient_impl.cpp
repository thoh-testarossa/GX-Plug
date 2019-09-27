//
// Created by liqi on 19-5-27.
//

#include "../UtilClient.cpp"
#include "../../algo/LabelPropagation/LabelPropagation.h"

template class UtilClient<double, double>;
template class UtilClient<std::pair<double, double>, std::pair<int, double>>;
template class UtilClient<LPA_Value, std::pair<int, int>>;
