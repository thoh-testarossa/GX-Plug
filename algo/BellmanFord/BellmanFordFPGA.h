//
// Created by Thoh Testarossa on 2019-03-12
//

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORDGPU_H
#define GRAPH_ALGO_BELLMANFORDGPU_H

#include "BellmanFord.h"
#include "../../util/xcl2/xcl2.hpp"

template<typename VertexValueType, typename MessageValueType>
class BellmanFordFPGA : public BellmanFord<VertexValueType, MessageValueType>
{
public:
    BellmanFordFPGA();
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

#endif //GRAPH_ALGO_BELLMANFORDGPU_H
