//
// Created by Thoh Testarossa on 2019-08-12.
//

#include "ConnectedComponentGPU.h"
//#include "kernel_src/BellmanFordGPU_kernel.h"

#include <iostream>
#include <algorithm>

template<typename VertexValueType>
ConnectedComponentGPU<VertexValueType>::ConnectedComponentGPU()
{

}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    ConnectedComponent<VertexValueType>::Init(vCount, eCount, numOfInitV);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;
}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                       const std::vector<int> &initVList)
{
    ConnectedComponent<VertexValueType>::GraphInit(g, activeVertices, initVList);
}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    ConnectedComponent<VertexValueType>::Deploy(vCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    this->vValueSet = new VertexValueType [vCount];
    err = cudaMalloc((void **)&this->d_vValueSet, vertexLimit * sizeof(int));

    this->mValueTable = new VertexValueType [vCount];
    err = cudaMalloc((void **)&this->d_mValueTable, vertexLimit * sizeof(int));

    err = cudaMalloc((void **)&this->d_vSet, vertexLimit * sizeof(Vertex));
    err = cudaMalloc((void **)&this->d_eGSet, ePerEdgeSet * sizeof(Edge));

    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);

    this->mInitVIndexSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mInitVIndexSet, mSize * sizeof(int));
    this->mDstSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mDstSet, mSize * sizeof(int));
    this->mValueSet = new VertexValueType [mSize];
    err = cudaMalloc((void **)&this->d_mValueSet, mSize * sizeof(double));
}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::Free()
{
    ConnectedComponent<VertexValueType>::Free();

    free(this->vValueSet);
    cudaFree(this->d_vValueSet);

    free(this->mValueTable);
    cudaFree(this->d_mValueTable);

    cudaFree(this->d_vSet);
    cudaFree(this->d_eGSet);

    free(this->mInitVIndexSet);
    cudaFree(this->d_mInitVIndexSet);
    free(this->mDstSet);
    cudaFree(this->d_mDstSet);
    free(this->mValueSet);
    cudaFree(this->d_mValueSet);
}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                      std::set<int> &activeVertices,
                                                      const MessageSet<VertexValueType> &mSet)
{

}

template<typename VertexValueType>
void
ConnectedComponentGPU<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                    const std::set<int> &activeVertices,
                                                    MessageSet<VertexValueType> &mSet)
{

}

template<typename VertexValueType>
void ConnectedComponentGPU<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV,
                                                            const int *initVSet, VertexValueType *vValues,
                                                            VertexValueType *mValues)
{

}

template<typename VertexValueType>
void
ConnectedComponentGPU<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet,
                                                          int numOfInitV, const int *initVSet,
                                                          const VertexValueType *vValues, VertexValueType *mValues)
{

}

