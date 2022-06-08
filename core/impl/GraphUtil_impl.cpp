//
// Created by Thoh Testarossa on 2019-05-25.
//

#include "../GraphUtil.cpp"
#include "../../algo/LabelPropagation/LabelPropagation.h"
#include "../../algo/PageRank/PageRank.h"
#include "../../algo/DDFS/DDFS.h"
#include "../../algo/BellmanFord/BellmanFord.h"
template class GraphUtil<double, double>;
template class GraphUtil<int, int>;
template class GraphUtil<std::pair<double, double>, PRA_MSG>;
template class GraphUtil<DFSValue, DFSMSG>;
template class GraphUtil<LPA_Value, LPA_MSG>;

template struct ComputeUnit<double>;
template struct ComputeUnit<int>;
template struct ComputeUnit<std::pair<double, double>>;
template struct ComputeUnit<DFSValue>;
template struct ComputeUnit<LPA_Value>;

template class ComputeUnitPackage<double>;
template class ComputeUnitPackage<int>;
template class ComputeUnitPackage<std::pair<double, double>>;
template class ComputeUnitPackage<DFSValue>;
template class ComputeUnitPackage<LPA_Value>;
