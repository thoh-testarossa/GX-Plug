#include "LabelPropagationGPU_kernel.h"

__global__ void MSGApply_kernel(Vertex *vSet, LPA_Value *vValues, int numOfMsg, LPA_MSG *mValueSet, int *offsetInValues)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		LPA_MSG msg = mValueSet[tid];
		int edgeIndex = msg.edgeOriginIndex;
		int destVId = msg.destVId;
		int label = msg.label;
		int offset = offsetInValues[destVId];

        //test
//        printf("edgeIndex: %d destVId: %d label: %d \n", edgeIndex, destVId, label);

		vValues[edgeIndex].destVId = destVId;
		vValues[edgeIndex].labelCnt = 1;
		vValues[edgeIndex].label = label;

	}
}

cudaError_t MSGApply_kernel_exec(Vertex *vSet, LPA_Value *vValues, int numOfMsg, LPA_MSG *mValueSet, int *offsetInValues)
{
	cudaError_t err = cudaSuccess;
	
	MSGApply_kernel<<<1, NUMOFGPUCORE>>>(vSet, vValues, numOfMsg, mValueSet, offsetInValues);
    err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGenMerge_kernel(LPA_MSG *mTransformdMergedMSGValueSet, Vertex *vSet, LPA_Value *vValues, int numOfEdge, Edge *eSet, int batchCnt)
{
	int tid = threadIdx.x;

	if(tid < numOfEdge)
	{
		int destVId = eSet[tid].dst;
		int srcVId = eSet[tid].src;
		int mValueIndex = (batchCnt << NUMOFGPUCORE_BIT) + tid;

		//test
//		printf("destvid %d srcvid %d label %d index %d\n", destVId, srcVId, vValues[srcVId].label, mValueIndex);

		mTransformdMergedMSGValueSet[mValueIndex].destVId = destVId;
		mTransformdMergedMSGValueSet[mValueIndex].edgeOriginIndex = eSet[tid].originIndex;
		mTransformdMergedMSGValueSet[mValueIndex].label = vValues[srcVId].label;
	}
}

cudaError_t MSGGenMerge_kernel_exec(LPA_MSG *mTransformdMergedMSGValueSet,
	Vertex *vSet, LPA_Value *vValues, int numOfEdge, Edge *eSet, int batchCnt)
{
	cudaError_t err = cudaSuccess;

	MSGGenMerge_kernel<<<1, NUMOFGPUCORE>>>(mTransformdMergedMSGValueSet, vSet, vValues, numOfEdge, eSet, batchCnt);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}