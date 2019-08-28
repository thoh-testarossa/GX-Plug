//
// Created by Thoh Testarossa on 2019-05-25.
//

#include "../GraphUtil.cpp"

#include "../../algo/DDFS/DDFS.h"

template class GraphUtil<double, double>;
template class GraphUtil<int, int>;
template class GraphUtil<std::pair<int, int>, std::pair<int, int>>;
template class GraphUtil<DFSValue, DFSMSG>;