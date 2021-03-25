#include "LabelPropagationGPU_kernel.h"

__global__ void MSGApply_kernel(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet)
{
    int tid = threadIdx.x;

    if (tid < numOfUnits)
    {
        if (!computeUnits[tid].destVertex.isMaster) return;

        computeUnits[tid].destValue.label = computeUnits[tid].srcValue.label;
        computeUnits[tid].destValue.destVId = computeUnits[tid].destVertex.vertexID;
    }
}

cudaError_t MSGApply_kernel_exec(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet)
{
    cudaError_t err = cudaSuccess;

    MSGApply_kernel << < 1, NUMOFGPUCORE >> > (numOfUnits, computeUnits, mValueSet);
    err = cudaGetLastError();

    cudaDeviceSynchronize();

    return err;
}

__global__ void MSGGenMerge_kernel(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet)
{
}

cudaError_t
MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<LPA_Value> *computeUnits, LPA_MSG *mValueSet)
{
    cudaError_t err = cudaSuccess;
    return err;
}