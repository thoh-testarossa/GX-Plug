#include "PageRankGPU_kernel.h"

__global__ void MSGApply_kernel(Vertex *vSet, double *vValues, int numOfMsg, int *mDstSet, PRA_MSG *mValueSet, double resetProb)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		int vID = mDstSet[tid];

		//test
		// printf("vId : %d value : %f\n", vID, mValueSet[tid].rank);

		vSet[vID].isActive = true;
		vSet[vID].needMerge = true;
		atomicAdd(&vValues[(vID << 1) + 1], (1.0 - resetProb) * mValueSet[tid].rank);
	}
}

cudaError_t MSGApply_kernel_exec(Vertex *vSet, double *vValues, int numOfMsg, int *mDstSet, PRA_MSG *mValueSet, double resetProb)
{
	cudaError_t err = cudaSuccess;
	
	MSGApply_kernel<<<1, NUMOFGPUCORE>>>(vSet, vValues, numOfMsg, mDstSet, mValueSet, resetProb);
    err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGenMerge_kernel(PRA_MSG *mTransformdMergedMSGValueSet,
	Vertex *vSet, double *vValues, int numOfEdge, Edge *eSet, int batchCnt, double threshold)
{
	int tid = threadIdx.x;

	if(tid < numOfEdge)
	{
		int srcVid = eSet[tid].src;

		int mValueIndex = (batchCnt << NUMOFGPUCORE_BIT) + tid;

		if(vValues[(srcVid << 1) + 1] > threshold)
		{
			//test
			// printf("msg - srcVid: %d destVid: %d\n", srcVid, eSet[tid].dst);
			// printf("vValue: %f weight %f\n", vValues[(srcVid << 1) + 1], eSet[tid].weight);
			// printf("mValueIndex = %d\n", mValueIndex);

			mTransformdMergedMSGValueSet[mValueIndex].destVId = eSet[tid].dst;
			mTransformdMergedMSGValueSet[mValueIndex].rank = vValues[(srcVid << 1) + 1] * eSet[tid].weight;
			
			//set the Active flag to clear the delta in vValue before array_apply op
			vSet[eSet[tid].dst].isActive = true;
		}
		else
		{
			//test
			// printf("nullmsg - srcVid: %d destVid: %d\n", srcVid, eSet[tid].dst);

			mTransformdMergedMSGValueSet[mValueIndex].destVId = -1;
			mTransformdMergedMSGValueSet[mValueIndex].rank = -1;
		}
	}
}

cudaError_t MSGGenMerge_kernel_exec(PRA_MSG *mTransformdMergedMSGValueSet,
	Vertex *vSet, double *vValues, int numOfEdge, Edge *eSet, int batchCnt, double threshold)
{
	cudaError_t err = cudaSuccess;

	MSGGenMerge_kernel<<<1, NUMOFGPUCORE>>>(mTransformdMergedMSGValueSet, vSet, vValues, numOfEdge, eSet, batchCnt, threshold);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}