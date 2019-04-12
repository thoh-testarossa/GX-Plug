#include "BellmanFordGPU_kernel.h"

__global__ void MSGApply_kernel(int numOfInitV, int *initVSet, double *vValues,
        int numOfMsg, int *mDstSet, int *mInitVSet, double *mValueSet,
        bool *AVCheckSet, int *initVIndexSet)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		int vID = mDstSet[tid];

		int vInitVIndex = initVIndexSet[mInitVSet[tid]];

		if(vInitVIndex != -1)
		{
			if(vValues[vID * numOfInitV + vInitVIndex] > mValueSet[tid])
			{
				vValues[vID * numOfInitV + vInitVIndex] = mValueSet[tid];
				AVCheckSet[vID] = true;
			}
		}
		else;
	}
}

cudaError_t MSGApply_kernel_exec(int numOfInitV, int *initVSet, double *vValues,
	int numOfMsg, int *mDstSet, int *mInitVSet, double *mValueSet,
	bool *AVCheckSet, int *initVIndexSet)
{
	cudaError_t err = cudaSuccess;
	
	MSGApply_kernel<<<1, NUMOFGPUCORE>>>(numOfInitV, initVSet, vValues, numOfMsg, mDstSet, mInitVSet, mValueSet, AVCheckSet, initVIndexSet);
    err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGen_kernel(int numOfEdge, bool *AVCheckSet, 
	int *eSrcSet, int *eDstSet, double *eWeightSet,
	int numOfInitV, int *initVSet, double *vValues,
	int *mDstSet, int *mInitVSet, double *mValueSet)
{
	int tid = threadIdx.x;

	if(tid < numOfEdge)
	{
		int vID = -1;

		if(AVCheckSet[eSrcSet[tid]] == true) vID = eSrcSet[tid];

		if(vID != -1)
		{
			for(int i = 0; i < numOfInitV; i++)
			{
				mInitVSet[tid * numOfInitV + i] = initVSet[i];
				mDstSet[tid * numOfInitV + i] = eDstSet[tid];
				mValueSet[tid * numOfInitV + i] = vValues[vID * numOfInitV + i] + eWeightSet[tid];
			}
		}
		else;
	}
}

cudaError_t MSGGen_kernel_exec(int numOfEdge, bool *AVCheckSet, 
	int *eSrcSet, int *eDstSet, double *eWeightSet,
	int numOfInitV, int *initVSet, double *vValues,
	int *mDstSet, int *mInitVSet, double *mValueSet)
{
	cudaError_t err = cudaSuccess;

	MSGGen_kernel<<<1, NUMOFGPUCORE>>>(numOfEdge, AVCheckSet, eSrcSet, eDstSet, eWeightSet, numOfInitV, initVSet, vValues, mDstSet, mInitVSet, mValueSet);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGMerge_kernel(unsigned long long int *mTransformdMergedMSGValueSet,
	int numOfInitV, int *initVSet, 
	int numOfMsg, int *mDstSet, int *mInitVSet, unsigned long long int *mValueTSet, 
	int *initVIndexSet)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		int vID = mDstSet[tid];
		int vInitVIndex = initVIndexSet[mInitVSet[tid]];

		if(vInitVIndex != -1)
		//Original mValue is needed to be changed to long long int form to execute atomic ops
			atomicMin(&mTransformdMergedMSGValueSet[vID * numOfInitV + vInitVIndex], mValueTSet[tid]);
	}
}

cudaError_t MSGMerge_kernel_exec(unsigned long long int *mTransformdMergedMSGValueSet, 
	int numOfInitV, int *initVSet, 
	int numOfMsg, int *mDstSet, int *mInitVSet, unsigned long long int *mValueTSet, 
	int *initVIndexSet)
{
	cudaError_t err = cudaSuccess;

	MSGMerge_kernel<<<1, NUMOFGPUCORE>>>(mTransformdMergedMSGValueSet, numOfInitV, initVSet, numOfMsg, mDstSet, mInitVSet, mValueTSet, initVIndexSet);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGenMerge_kernel(unsigned long long int *mTransformdMergedMSGValueSet,
	bool *AVCheckSet, int numOfInitV, int *initVSet, double *vValues,
	int numOfEdge, Edge *eSet)
{
	int tid = threadIdx.x;

	if(tid < numOfEdge)
	{
		int vID = -1;
		if(AVCheckSet[eSet[tid].src] == true) vID = eSet[tid].dst;

		if(vID != -1)
		{
			for(int i = 0; i < numOfInitV; i++)
				atomicMin(&mTransformdMergedMSGValueSet[vID * numOfInitV + i], __double_as_longlong(vValues[eSet[tid].src * numOfInitV + i] + eSet[tid].weight));
		}
		else;
	}
}

cudaError_t MSGGenMerge_kernel_exec(unsigned long long int *mTransformdMergedMSGValueSet,
	bool *AVCheckSet, int numOfInitV, int *initVSet, double *vValues,
	int numOfEdge, Edge *eSet)
{
	cudaError_t err = cudaSuccess;

	MSGGenMerge_kernel<<<1, NUMOFGPUCORE>>>(mTransformdMergedMSGValueSet, AVCheckSet, numOfInitV, initVSet, vValues, numOfEdge, eSet);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}