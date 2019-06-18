//
// Created by Thoh Testarossa on 2019-03-12.
//

#include "BellmanFordGPU.h"
#include "kernel_src/BellmanFordGPU_kernel.h"

#include <iostream>
#include <algorithm>

#define NULLMSG -1

//Transformation function
//This two function is the parts of MSGMerge for adapting CUDA's atomic ops
//double -> long long int
unsigned long long int doubleAsLongLongInt(double a)
{
    unsigned long long int *ptr = (unsigned long long int *)&a;
    return *ptr;
}
//long long int -> double
double longLongIntAsDouble(unsigned long long int a)
{
    double *ptr = (double *)&a;
    return *ptr;
}
//Transformation functions end

template <typename VertexValueType>
BellmanFordGPU<VertexValueType>::BellmanFordGPU()
{
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::Init(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    BellmanFord<VertexValueType>::Init(g, activeVertices, initVList);
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::Deploy(int vCount, int numOfInitV)
{
    BellmanFord<VertexValueType>::Deploy(vCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    this->mPerMSGSet = NUMOFGPUCORE;
    this->ePerEdgeSet = NUMOFGPUCORE;

    this->initVSet = new int [numOfInitV];
    err = cudaMalloc((void **)&this->d_initVSet, this->numOfInitV * sizeof(int));
    this->initVIndexSet = new int [vCount];
    err = cudaMalloc((void **)&this->d_initVIndexSet, vCount * sizeof(int));
    this->vValueSet = new double [vCount * this->numOfInitV];
    err = cudaMalloc((void **)&this->d_vValueSet, vCount * this->numOfInitV * sizeof(double));

    this->mValueTable = new VertexValueType [vCount * this->numOfInitV];

    this->AVCheckSet = new bool [vCount];
    err = cudaMalloc((void **)&this->d_AVCheckSet, vCount * sizeof(bool));

    this->eSrcSet = new int [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eSrcSet, ePerEdgeSet * sizeof(int));
    this->eDstSet = new int [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eDstSet, ePerEdgeSet * sizeof(int));
    this->eWeightSet = new double [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eWeightSet, ePerEdgeSet * sizeof(double));

    err = cudaMalloc((void **)&this->d_vSet, vCount * sizeof(Vertex));
    err = cudaMalloc((void **)&this->d_eGSet, ePerEdgeSet * sizeof(Edge));

    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);

    this->mInitVSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mInitVSet, mSize * sizeof(int));
    this->mDstSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mDstSet, mSize * sizeof(int));
    this->mValueSet = new double [mSize];
    err = cudaMalloc((void **)&this->d_mValueSet, mSize * sizeof(double));

    this->activeVerticesSet = new int [vCount];
    err = cudaMalloc((void **)&this->d_activeVerticesSet, vCount * sizeof(int));

    this->mMergedMSGValueSet = new VertexValueType [vCount * numOfInitV];
    this->mTransformedMergedMSGValueSet = new unsigned long long int [vCount * numOfInitV];
    err = cudaMalloc((void **)&d_mTransformedMergedMSGValueSet, numOfInitV * vCount * sizeof(unsigned long long int));

    this->mValueTSet = new unsigned long long int [mPerMSGSet];
    err = cudaMalloc((void **)&this->d_mValueTSet, mPerMSGSet * sizeof(unsigned long long int));
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::Free()
{
    BellmanFord<VertexValueType>::Free();

    free(this->initVSet);
    cudaFree(this->d_initVSet);
    free(this->initVIndexSet);
    cudaFree(this->d_initVIndexSet);
    free(this->vValueSet);
    cudaFree(this->d_vValueSet);

    free(this->mValueTable);

    free(this->AVCheckSet);
    cudaFree(this->d_AVCheckSet);

    free(this->eSrcSet);
    cudaFree(this->d_eSrcSet);
    free(this->eDstSet);
    cudaFree(this->d_eDstSet);
    free(this->eWeightSet);
    cudaFree(this->d_eWeightSet);

    cudaFree(this->d_vSet);
    cudaFree(this->d_eGSet);

    free(this->mInitVSet);
    cudaFree(this->d_mInitVSet);
    free(this->mDstSet);
    cudaFree(this->d_mDstSet);
    free(this->mValueSet);
    cudaFree(this->d_mValueSet);

    free(this->activeVerticesSet);
    cudaFree(this->d_activeVerticesSet);

    free(this->mMergedMSGValueSet);
    free(this->mTransformedMergedMSGValueSet);
    cudaFree(this->d_mTransformedMergedMSGValueSet);

    free(this->mValueTSet);
    cudaFree(this->d_mValueTSet);
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<VertexValueType> &mSet)
{
    //Availability check
    if(g.vCount == 0) return;

    //AVCheckSet Init
    for(int i = 0; i < g.vCount; i++) this->AVCheckSet[i] = false;

    //MSG Init
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        this->mValueTable[i] = (VertexValueType)INVALID_MASSAGE;
    for(int i = 0; i < mSet.mSet.size(); i++)
    {
        auto &mv = this->mValueTable[mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex];
        if(mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }

    //array form computation
    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], this->mValueTable);

    //Active vertices set assembly
    activeVertices.clear();
    for(int i = 0; i < g.vCount; i++)
    {
        if(g.vList.at(i).isActive)
            activeVertices.insert(i);
    }
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<VertexValueType> &mSet)
{
    //Generate merged msgs directly

    //Availability check
    if(g.vCount == 0) return;

    //array form computation
    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], this->mMergedMSGValueSet);

    //Package mMergedMSGValueSet to result mSet
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        if((double)this->mMergedMSGValueSet[i] != INVALID_MASSAGE)
        {
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<VertexValueType>(initV, dst, this->mMergedMSGValueSet[i]));
        }
    }
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues)
{
    //Availability check
    if(vCount == 0) return;

    //CUDA init
    cudaError_t err = cudaSuccess;

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    err = cudaMemcpy(this->d_vValueSet, (double *)vValues, vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //vSet Init
    //AVCheck
    for(int i = 0; i < vCount; i++) vSet[i].isActive = false;
    err = cudaMemcpy(this->d_vSet, vSet, vCount * sizeof(Vertex), cudaMemcpyHostToDevice);

    //Apply msgs to v
    int mGCount = 0;
    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(mValues[i] != INVALID_MASSAGE) //Adding msgs to batches
        {
            this->mInitVSet[mGCount] = initVSet[i % numOfInitV];
            this->mDstSet[mGCount] = i / numOfInitV;
            this->mValueSet[mGCount] = (double)mValues[i];
            mGCount++;
        }
        if(mGCount == this->mPerMSGSet || i == vCount * numOfInitV - 1) //A batch of msgs will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_mInitVSet, this->mInitVSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, this->mValueSet, mGCount * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGApply_kernel_exec(this->d_vSet, numOfInitV, this->d_initVSet, this->d_vValueSet, mGCount, this->d_mDstSet, this->d_mInitVSet, this->d_mValueSet);

            mGCount = 0;
        }
    }

    //Re-package the data

    //Memory copy back
    err = cudaMemcpy(vSet, this->d_vSet, vCount * sizeof(Vertex), cudaMemcpyDeviceToHost);
    err = cudaMemcpy((double *)vValues, this->d_vValueSet, vCount * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues)
{
    //Generate merged msgs directly

    //Availability check
    if(vCount == 0) return;

    //Memory allocation
    cudaError_t err = cudaSuccess;

    //vSet Init
    err = cudaMemcpy(this->d_vSet, vSet, vCount * sizeof(Vertex), cudaMemcpyHostToDevice);

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    err = cudaMemcpy(this->d_vValueSet, (double *)vValues, vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //mMergedMSGValueSet Init
    for(int i = 0; i < vCount * numOfInitV; i++)
        mValues[i] = (VertexValueType)INVALID_MASSAGE;

    //Transform to the long long int form which CUDA can do atomic ops
    //unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for(int i = 0; i < vCount * numOfInitV; i++)
        this->mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt((double)mValues[i]);
    
    err = cudaMemcpy(this->d_mTransformedMergedMSGValueSet, this->mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    //e batch processing
    int eGCount = 0;

    std::vector<Edge> eGSet = std::vector<Edge>();
    eGSet.reserve(this->ePerEdgeSet);

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive) //Add es to batches
        {
            for(int j = 0; j < numOfInitV; j++)
            {
                if(vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight < vValues[eSet[i].dst * numOfInitV + j])
                {
                    eGSet.emplace_back(eSet[i]);
                    eGCount++;
                    break;
                }
            }
        }
        if(eGCount == this->ePerEdgeSet || i == eCount - 1) //A batch of es will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_eGSet, &eGSet[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGGenMerge_kernel_exec(this->d_mTransformedMergedMSGValueSet, this->d_vSet, numOfInitV, this->d_initVSet, this->d_vValueSet, eGCount, this->d_eGSet);

            eGCount = 0;
            eGSet.clear();
        }
    }

    //Re-package the data
    //Memory copy back
    err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    //Transform back to original double form
    for(int i = 0; i < vCount * numOfInitV; i++)
        mValues[i] = (VertexValueType)(longLongIntAsDouble(this->mTransformedMergedMSGValueSet[i]));
}