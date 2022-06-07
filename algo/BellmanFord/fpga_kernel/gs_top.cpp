// #include "noStreamLib.h"
#include "config.h"
extern "C" {
	void gs_top(
			int* srcValue,
			int* ewValue,
			int* dstArray,
			int* dstValue,
			int* activeNode,
			int edgeNum
			)
	{
#pragma HLS INTERFACE m_axi port=srcValue offset=slave bundle=gmem0 
#pragma HLS INTERFACE s_axilite port=srcValue bundle=control

#pragma HLS INTERFACE m_axi port=ewValue offset=slave bundle=gmem1 
#pragma HLS INTERFACE s_axilite port=ewValue bundle=control

#pragma HLS INTERFACE m_axi port=dstArray offset=slave bundle=gmem2 
#pragma HLS INTERFACE s_axilite port=dstArray bundle=control

#pragma HLS INTERFACE m_axi port=dstValue offset=slave bundle=gmem3 
#pragma HLS INTERFACE s_axilite port=dstValue bundle=control

#pragma HLS INTERFACE m_axi port=activeNode offset=slave bundle=gmem4 
#pragma HLS INTERFACE s_axilite port=activeNode bundle=control

#pragma HLS INTERFACE s_axilite port=edgeNum       bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

	int VList[MAX_READ_IN_ONE_LOOP];
	int dstList[MAX_READ_IN_ONE_LOOP];
	int valuePool[MAX_READ_IN_ONE_LOOP];
	int activeState[MAX_READ_IN_ONE_LOOP];
	int activeNodeNum = 0;
	int eCount = edgeNum;
	int loop = ((eCount - 1) / MAX_READ_IN_ONE_LOOP) + 1;
	for(int i = 0 ; i < loop ; i++)
	{	
		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ; j++)
		{
			VList[j] = srcValue[i*MAX_READ_IN_ONE_LOOP + j] + ewValue[i*MAX_READ_IN_ONE_LOOP + j]; 
		}

		
		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ;j++)
		{
			dstList[j] = dstArray[i*MAX_READ_IN_ONE_LOOP + j];
		}

		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ;j++)
		{
			valuePool[j] = dstValue[i*MAX_READ_IN_ONE_LOOP + j];
		}

		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ;j++)
		{
			if(VList[j] < valuePool[j])
			{
				valuePool[i*MAX_READ_IN_ONE_LOOP + j] = VList[j];
				activeState[i*MAX_READ_IN_ONE_LOOP + j] = 1;
				activeNodeNum++;
			}	
		}

		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ;j++)
		{
			dstValue[i*MAX_READ_IN_ONE_LOOP + j] = valuePool[j];			
		}

		for(int j = 0 ; j < MAX_READ_IN_ONE_LOOP ;j++)
		{
			activeNode[i*MAX_READ_IN_ONE_LOOP + j] = activeState[j];
		}

	}
	dstValue[edgeNum] = activeNodeNum;
 	}
}
