#include "BellmanFordGPU_kernel.h"

__global__ void MSGApply_kernel(int numOfUnits, ComputeUnit<double> *computeUnits, double *mValueSet, int numOfInitV)
{
    int tid = threadIdx.x;

    if (tid < numOfUnits)
    {

        int vID = computeUnits[tid].destVertex.vertexID;
        int originIndex = computeUnits[tid].indexOfInitV;

        if (!computeUnits[tid].destVertex.isMaster) return;
        if (computeUnits[tid].destValue > mValueSet[vID * numOfInitV + originIndex])
        {
            computeUnits[tid].destValue = mValueSet[vID * numOfInitV + originIndex];
            computeUnits[tid].destVertex.isActive = true;
        }
    }
}

cudaError_t MSGApply_kernel_exec(int numOfUnits, ComputeUnit<double> *computeUnits, double *mValueSet, int numOfInitV)
{
    cudaError_t err = cudaSuccess;

    MSGApply_kernel<<< 1, NUMOFGPUCORE >>>(numOfUnits, computeUnits, mValueSet, numOfInitV);
    err = cudaGetLastError();

    cudaDeviceSynchronize();

    return err;
}

__global__ void MSGGenMerge_kernel(int numOfUnits, ComputeUnit<double> *computeUnits, unsigned long long int *mValueSet,
                                   int numOfInitV)
{
    int tid = threadIdx.x;

    if (tid < numOfUnits)
    {
        int vID = computeUnits[tid].destVertex.vertexID;
        int originIndex = computeUnits[tid].indexOfInitV;
        computeUnits[tid].destVertex.isActive = false;
        computeUnits[tid].srcVertex.isActive = false;

        atomicMin(&mValueSet[vID * numOfInitV + originIndex],
                  __double_as_longlong(computeUnits[tid].srcValue + computeUnits[tid].edgeWeight));
    }
}

cudaError_t
MSGGenMerge_kernel_exec(int numOfUnits, ComputeUnit<double> *computeUnits, unsigned long long int *mValueSet,
                        int numOfInitV)
{
    cudaError_t err = cudaSuccess;

    MSGGenMerge_kernel<<< 1, NUMOFGPUCORE >>>(numOfUnits, computeUnits, mValueSet, numOfInitV);
    err = cudaGetLastError();

    cudaDeviceSynchronize();

    return err;
}