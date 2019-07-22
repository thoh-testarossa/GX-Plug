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

//Internal method for different GPU copy situations in BF algo
template <typename VertexValueType>
auto BellmanFordGPU<VertexValueType>::MSGGenMerge_GPU_MVCopy(Vertex *d_vSet, const Vertex *vSet,
                                   double *d_vValues, const double *vValues,
                                   unsigned long long int *d_mTransformedMergedMSGValueSet,
                                   unsigned long long int *mTransformedMergedMSGValueSet,
                                   int vGCount, int numOfInitV)
{
    auto err = cudaSuccess;

    //vSet copy
    err = cudaMemcpy(d_vSet, vSet, vGCount * sizeof(Vertex), cudaMemcpyHostToDevice);

    //vValueSet copy
    err = cudaMemcpy(d_vValues, vValues, vGCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //Transform to the long long int form which CUDA can do atomic ops
    //unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for (int i = 0; i < vGCount * numOfInitV; i++)
        mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt((double) INVALID_MASSAGE);

    //mTransformedMergedMSGValueSet copy
    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mTransformedMergedMSGValueSet,
                     vGCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    return err;
}

template <typename VertexValueType>
auto BellmanFordGPU<VertexValueType>::MSGApply_GPU_VVCopy(Vertex *d_vSet, const Vertex *vSet,
                                double *d_vValues, const double *vValues,
                                int vGCount, int numOfInitV)
{
    auto err = cudaSuccess;

    //vSet copy
    err = cudaMemcpy(d_vSet, vSet, vGCount * sizeof(Vertex), cudaMemcpyHostToDevice);

    //vValueSet copy
    err = cudaMemcpy(d_vValues, vValues, vGCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    return err;
}

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

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;

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

    err = cudaMalloc((void **)&this->d_vSet, vertexLimit * sizeof(Vertex));
    err = cudaMalloc((void **)&this->d_eGSet, ePerEdgeSet * sizeof(Edge));

    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);

    this->mInitVIndexSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mInitVIndexSet, mSize * sizeof(int));
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

    free(this->mInitVIndexSet);
    cudaFree(this->d_mInitVIndexSet);
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

    bool needReflect = vCount > VERTEXSCALEINGPU;

    //AVCheck
    for (int i = 0; i < vCount; i++) vSet[i].isActive = false;

    if(!needReflect)
    {
        err = MSGApply_GPU_VVCopy(this->d_vSet, vSet,
                            this->d_vValueSet, (double *)vValues,
                            vCount, numOfInitV);

    }

    //Apply msgs to v
    int mGCount = 0;
    auto mGSet = MessageSet<VertexValueType>();

    auto r_mGSet = MessageSet<VertexValueType>();
    auto r_vSet = std::vector<Vertex>();
    auto r_vValueSet = std::vector<VertexValueType>();

    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(mValues[i] != INVALID_MASSAGE) //Adding msgs to batchs
        {
            mGSet.insertMsg(Message<VertexValueType>(initVSet[i % numOfInitV], i / numOfInitV, mValues[i]));
            mGCount++;
        }
        if(mGCount == this->mPerMSGSet || i == vCount * numOfInitV - 1) //A batch of msgs will be transferred into GPU. Don't forget last batch!
        {
            auto reflectIndex = std::vector<int>();
            auto reversedIndex = std::vector<int>();

            //Reflection for message & vertex & vValues
            if(needReflect)
            {
                //MSG reflection
                r_mGSet = this->reflectM(mGSet, vCount, reflectIndex, reversedIndex);

                for(int i = 0; i < r_mGSet.mSet.size(); i++)
                {
                    this->mInitVIndexSet[i] = vSet[r_mGSet.mSet.at(i).src].initVIndex;
                    this->mDstSet[i] = r_mGSet.mSet.at(i).dst;
                    this->mValueSet[i] = (double)r_mGSet.mSet.at(i).value;
                }

                //v reflection
                r_vSet.clear();
                for(int i = 0; i < reflectIndex.size(); i++)
                    r_vSet.emplace_back(i, false, vSet[reflectIndex.at(i)].initVIndex);

                //vValue reflection
                r_vValueSet.clear();
                r_vValueSet.reserve(mPerMSGSet * numOfInitV);
                r_vValueSet.assign(mPerMSGSet * numOfInitV, INT32_MAX >> 1);
                for(int i = 0; i < reflectIndex.size(); i++)
                {
                    for(int j = 0; j < numOfInitV; j++)
                        r_vValueSet.at(i * numOfInitV + j) = vValues[reflectIndex[i] * numOfInitV + j];
                }

                //vSet & vValueSet Init
                err = MSGApply_GPU_VVCopy(d_vSet, &r_vSet[0],
                                    d_vValueSet, (double *)&r_vValueSet[0],
                                    reflectIndex.size(), numOfInitV);
            }
            else
            {
                //Use original msg
                for(int i = 0; i < mGSet.mSet.size(); i++)
                {
                    this->mInitVIndexSet[i] = vSet[mGSet.mSet.at(i).src].initVIndex;
                    this->mDstSet[i] = mGSet.mSet.at(i).dst;
                    this->mValueSet[i] = (double)mGSet.mSet.at(i).value;
                }
            }

            //MSG memory copy
            err = cudaMemcpy(this->d_mInitVIndexSet, this->mInitVIndexSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, this->mValueSet, mGCount * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            for(int i = 0; i < mGCount; i += NUMOFGPUCORE)
            {
                int msgNumUsedForExec = (mGCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (mGCount - i);

                err = MSGApply_kernel_exec(this->d_vSet, numOfInitV, this->d_initVSet, this->d_vValueSet, msgNumUsedForExec,
                                           &this->d_mDstSet[i], &this->d_mInitVIndexSet[i], &this->d_mValueSet[i]);
            }

            //Deflection
            if(needReflect)
            {
                err = cudaMemcpy(&r_vSet[0], this->d_vSet, reflectIndex.size() * sizeof(Vertex), cudaMemcpyDeviceToHost);
                err = cudaMemcpy((double *)&r_vValueSet[0], this->d_vValueSet, reflectIndex.size() * numOfInitV * sizeof(double),
                                 cudaMemcpyDeviceToHost);

                for(int i = 0; i < reflectIndex.size(); i++)
                {
                    vSet[reflectIndex[i]] = r_vSet[i];
                    for(int j = 0; j < numOfInitV; j++)
                        vValues[reflectIndex[i] * numOfInitV + j] = r_vValueSet[i * numOfInitV + j];
                }
            }

            mGSet.mSet.clear();
            mGCount = 0;
        }
    }

    //Re-package the data

    //Memory copy back
    if(!needReflect)
    {
        err = cudaMemcpy(vSet, this->d_vSet, vCount * sizeof(Vertex), cudaMemcpyDeviceToHost);
        err = cudaMemcpy((double *)vValues, this->d_vValueSet, vCount * numOfInitV * sizeof(double),
                         cudaMemcpyDeviceToHost);
    }
}

template <typename VertexValueType>
void BellmanFordGPU<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues)
{
    //Generate merged msgs directly

    //Availability check
    if(vCount == 0) return;

    //Memory allocation
    cudaError_t err = cudaSuccess;

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //Graph scale check
    bool needReflect = vCount > VERTEXSCALEINGPU;

    if(!needReflect)
        err = MSGGenMerge_GPU_MVCopy(this->d_vSet, vSet,
                               this->d_vValueSet, (double *)vValues,
                               this->d_mTransformedMergedMSGValueSet,
                               this->mTransformedMergedMSGValueSet,
                               vCount, numOfInitV);

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
            auto reflectIndex = std::vector<int>();
            auto reversedIndex = std::vector<int>();

            auto r_g = Graph<VertexValueType>(0);

            //Reflection
            if(needReflect)
            {
                bool *tmp_AVCheckList = new bool [vCount];
                for(int i = 0; i < vCount; i++) tmp_AVCheckList[i] = vSet[i].isActive;

                auto tmp_o_g = Graph<VertexValueType>(vCount, 0, numOfInitV, initVSet, nullptr, nullptr, nullptr, AVCheckSet);

                r_g = this->reflectG(tmp_o_g, eGSet, reflectIndex, reversedIndex);

                err = MSGGenMerge_GPU_MVCopy(this->d_vSet, &r_g.vList[0],
                                             this->d_vValueSet, (double *)&r_g.verticesValue[0],
                                             this->d_mTransformedMergedMSGValueSet,
                                             this->mTransformedMergedMSGValueSet,
                                             r_g.vCount, numOfInitV);

                err = cudaMemcpy(this->d_eGSet, &r_g.eList[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);
            }
            else
                err = cudaMemcpy(this->d_eGSet, &eGSet[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);

            //Kernel Execution (no matter whether g is reflected or not)
            for(int i = 0; i < eGCount; i += NUMOFGPUCORE)
            {
                int edgeNumUsedForExec = (eGCount - i > NUMOFGPUCORE) ? NUMOFGPUCORE : (eGCount - i);

                err = MSGGenMerge_kernel_exec(this->d_mTransformedMergedMSGValueSet, this->d_vSet, numOfInitV,
                                              this->d_initVSet, this->d_vValueSet, edgeNumUsedForExec, &this->d_eGSet[i]);
            }

            //Deflection
            if(needReflect)
            {
                //Re-package the data
                //Memory copy back
                err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet,
                                 r_g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

                //Transform back to original double form (deflection)
                for (int i = 0; i < r_g.vCount * numOfInitV; i++)
                    mValues[reflectIndex[i / numOfInitV] * numOfInitV + i % numOfInitV] =
                            (VertexValueType) (longLongIntAsDouble(this->mTransformedMergedMSGValueSet[i]));
            }
            else;

            eGCount = 0;
            eGSet.clear();
        }
    }

    if(!needReflect)
    {
        //Re-package the data
        //Memory copy back
        err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet,
                         vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        //Transform back to original double form
        for (int i = 0; i < vCount * numOfInitV; i++)
            mValues[i] = (VertexValueType) (longLongIntAsDouble(this->mTransformedMergedMSGValueSet[i]));
    }
}
