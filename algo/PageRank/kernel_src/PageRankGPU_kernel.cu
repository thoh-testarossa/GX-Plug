#include "PageRankGPU_kernel.h"

__global__ void
MSGApply_kernel(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet)
{
    int tid = threadIdx.x;

    if (tid < numOfUnits)
    {
        int destVId = computeUnits[tid].destVertex.vertexID;

        computeUnits[tid].srcVertex.isActive = false;
        computeUnits[tid].destVertex.isActive = false;

        if (mValueSet[destVId].destVId == -1 || !computeUnits[tid].destVertex.isMaster) return;

        computeUnits[tid].destVertex.isActive = true;
        computeUnits[tid].destValue.first += mValueSet[destVId].rank;
        computeUnits[tid].destValue.second = mValueSet[destVId].rank;
    }
}

cudaError_t
MSGApply_kernel_exec(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet)
{
    cudaError_t err = cudaSuccess;

    MSGApply_kernel << < 1, NUMOFGPUCORE >> > (numOfUnits, computeUnits, mValueSet);
    err = cudaGetLastError();

    cudaDeviceSynchronize();

    return err;
}

__global__ void
MSGGenMerge_kernel(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet,
                   double resetProb, double deltaThreshold)
{
    int tid = threadIdx.x;

    if (tid < numOfUnits)
    {
        int destVId = computeUnits[tid].destVertex.vertexID;

        if (computeUnits[tid].srcValue.second > deltaThreshold)
        {
            mValueSet[destVId].destVId = destVId;
            atomicAdd(&mValueSet[destVId].rank,
                      computeUnits[tid].srcValue.second * computeUnits[tid].edgeWeight * (1.0 - resetProb));
        }
    }
}

cudaError_t
MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<std::pair<double, double>> *computeUnits, PRA_MSG *mValueSet,
                        double resetProb, double deltaThreshold)
{
    cudaError_t err = cudaSuccess;

    MSGGenMerge_kernel << < 1, NUMOFGPUCORE >> >
                               (numOfUnits, computeUnits, mValueSet, resetProb, deltaThreshold);
    err = cudaGetLastError();

    cudaDeviceSynchronize();

    return err;
}