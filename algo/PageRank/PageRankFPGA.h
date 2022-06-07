#pragma once

#ifndef GRAPH_ALGO_PAGERANKFPGA_H
#define GRAPH_ALGO_PAGERANKFPGA_H
#include "PageRank.h"
#include "../../util/xcl2/xcl2.hpp"

template<typename VertexValueType, typename MessageValueType>
class PageRankFPGA : public PageRank<VertexValueType, MessageValueType>
{
public:
    PageRankFPGA();
    void InitFPGAEnv ();

    int MSGApply_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                       MessageValueType *mValues) override;

    int MSGGenMerge_array(int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,
                          MessageValueType *mValues) override;

protected:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel krnl;
};

#endif 
