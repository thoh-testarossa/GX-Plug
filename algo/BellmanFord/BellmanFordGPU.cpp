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

BellmanFordGPU::BellmanFordGPU()
{
}

void BellmanFordGPU::Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
{
    BellmanFord::Init(g, activeVertice, initVList);
}

void BellmanFordGPU::Deploy(int vCount, int numOfInitV)
{
    BellmanFord::Deploy(vCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    this->mPerMSGSet = NUMOFGPUCORE;
    this->ePerEdgeSet = NUMOFGPUCORE;

    this->initVSet = new int [numOfInitV];
    err = cudaMalloc((void **)&this->d_initVSet, this->numOfInitV * sizeof(int));

    this->vValueSet = new double [vCount * this->numOfInitV];
    err = cudaMalloc((void **)&this->d_vValueSet, vCount * this->numOfInitV * sizeof(double));

    this->AVCheckSet = new bool [vCount];
    err = cudaMalloc((void **)&this->d_AVCheckSet, vCount * sizeof(bool));

    this->eSrcSet = new int [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eSrcSet, ePerEdgeSet * sizeof(int));
    this->eDstSet = new int [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eDstSet, ePerEdgeSet * sizeof(int));
    this->eWeightSet = new double [ePerEdgeSet];
    err = cudaMalloc((void **)&this->d_eWeightSet, ePerEdgeSet * sizeof(double));

    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);

    this->mInitVSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mInitVSet, mSize * sizeof(int));
    this->mDstSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mDstSet, mSize * sizeof(int));
    this->mValueSet = new double [mSize];
    err = cudaMalloc((void **)&this->d_mValueSet, mSize * sizeof(double));

    this->activeVerticeSet = new int [vCount];
    err = cudaMalloc((void **)&this->d_activeVerticeSet, vCount * sizeof(int));

    this->mMergedMSGValueSet = new double [vCount * numOfInitV];
    this->mTransformedMergedMSGValueSet = new unsigned long long int [vCount * numOfInitV];
    err = cudaMalloc((void **)&d_mTransformedMergedMSGValueSet, numOfInitV * vCount * sizeof(unsigned long long int));

    this->mValueTSet = new unsigned long long int [mPerMSGSet];
    err = cudaMalloc((void **)&this->d_mValueTSet, mPerMSGSet * sizeof(unsigned long long int));
}

void BellmanFordGPU::Free()
{
    BellmanFord::Free();

    free(this->initVSet);
    cudaFree(this->d_initVSet);

    free(this->vValueSet);
    cudaFree(this->d_vValueSet);

    free(this->AVCheckSet);
    cudaFree(this->d_AVCheckSet);

    free(this->eSrcSet);
    cudaFree(this->d_eSrcSet);
    free(this->eDstSet);
    cudaFree(this->d_eDstSet);
    free(this->eWeightSet);
    cudaFree(this->d_eWeightSet);

    free(this->mInitVSet);
    cudaFree(this->d_mInitVSet);
    free(this->mDstSet);
    cudaFree(this->d_mDstSet);
    free(this->mValueSet);
    cudaFree(this->d_mValueSet);

    free(this->activeVerticeSet);
    cudaFree(this->d_activeVerticeSet);

    free(this->mMergedMSGValueSet);
    free(this->mTransformedMergedMSGValueSet);
    cudaFree(this->d_mTransformedMergedMSGValueSet);

    free(this->mValueTSet);
    cudaFree(this->d_mValueTSet);
}

void BellmanFordGPU::MSGApply(Graph &g, std::set<int> &activeVertice, const MessageSet &mSet)
{
    //Availability check
    if(g.vCount == 0) return;

    //Organise m group
    int numG = mSet.mSet.size() / mPerMSGSet + 1;
    std::vector<MessageSet> mSetG = std::vector<MessageSet>();
    for(int i = 0; i < numG; i++) mSetG.push_back(MessageSet());
    for(int i = 0; i < mSet.mSet.size(); i++) mSetG.at(i / mPerMSGSet).insertMsg(mSet.mSet.at(i));

    //Init
    int counterForIter = 0;

    //Memory allocation
    cudaError_t err = cudaSuccess;
    
    //initVSet Init
    counterForIter = 0;
    for(auto iter = g.vList.at(0).value.begin(); iter != g.vList.at(0).value.end() && counterForIter < numOfInitV; iter++)
    {
        initVSet[counterForIter] = iter->first;
        counterForIter++;
    }

    err = cudaMemcpy(d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    for(int i = 0; i < g.vCount; i++)
    {
        counterForIter = 0;
        for(auto iter = g.vList.at(i).value.begin(); iter != g.vList.at(i).value.end(); iter++)
        {
            vValueSet[i * numOfInitV + counterForIter] = iter->second;
            counterForIter++;
        }
    }

    err = cudaMemcpy(d_vValueSet, vValueSet, g.vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //AVCheckSet Init
    for(int i = 0; i < g.vCount; i++) AVCheckSet[i] = false;

    err = cudaMemcpy(d_AVCheckSet, AVCheckSet, g.vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //Loop by groups of msg
    for(auto mS : mSetG)
    {
        //Transfer data into GPU
        //info necessary: numOfInitV, initVSet, values of v for initV, a group of msg
        //size: 1 + numOfInitV + vCount * numOfInitV + msgGroup.size()

        //msg info Set
        for(int i = 0; i < mS.mSet.size(); i++)
        {
            mInitVSet[i] = mS.mSet.at(i).src;
            mDstSet[i] = mS.mSet.at(i).dst;
            mValueSet[i] = mS.mSet.at(i).value;
        }

        //GPU kernel function (one GPU thread handles a msg)
        //How to execute it parallel without IO conflict?
        //Actually, msg in mSet here are all not-conflict since they're the msgs combined by MSGMerge before

        //Memory copy
        err = cudaMemcpy(d_mInitVSet, mInitVSet, mS.mSet.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mDstSet, mDstSet, mS.mSet.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mValueSet, mValueSet, mS.mSet.size() * sizeof(double), cudaMemcpyHostToDevice);

        //Kernel Execution
        err = MSGApply_kernel_exec(numOfInitV, d_initVSet, d_vValueSet, mS.mSet.size(), d_mDstSet, d_mInitVSet, d_mValueSet, d_AVCheckSet);
    }

    //Re-package the data
    //info necessary: AVCheck table, values of v for initV
    //Size: vCount * (1 + numOfInitV)

    //Memory copyback
    err = cudaMemcpy(AVCheckSet, d_AVCheckSet, g.vCount * sizeof(bool), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(vValueSet, d_vValueSet, g.vCount * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    //Active vertice set assembly
    for(int i = 0; i < g.vCount; i++)
    {
        if(AVCheckSet[i])
            activeVertice.insert(i);
    }

    //g assembly
    for(int i = 0; i < g.vCount; i++)
    {
        g.vList.at(i).isActive = AVCheckSet[i];

        counterForIter = 0;
        for(auto iter = g.vList.at(i).value.begin(); iter != g.vList.at(i).value.end(); iter++)
        {
            if(vValueSet[i * numOfInitV + counterForIter] < iter->second)
                iter->second = vValueSet[i * numOfInitV + counterForIter];
            counterForIter++;
        }
    }
}

void BellmanFordGPU::MSGGen(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet)
{
    //Availability check
    if(g.vCount == 0) return;

    //Organise e group
    int ePerEdgeSet = NUMOFGPUCORE;
    int numG = g.eCount / ePerEdgeSet + 1;
    auto eSetG = std::vector<std::vector<Edge>>();
    for(int i = 0; i < numG; i++) eSetG.push_back(std::vector<Edge>());

    //Filter of valid message through isActive in source vertex
    for(int i = 0, validEdgeNum = 0; i < g.eCount; i++)
    {
        if(g.vList.at(g.eList.at(i).src).isActive)
            eSetG.at(validEdgeNum++ / ePerEdgeSet).push_back(g.eList.at(i));
    }

    //Init
    int counterForIter = 0;

    //Memory allocation
    cudaError_t err = cudaSuccess;
    
    //AVCheckSet Init
    for(int i = 0; i < g.vCount; i++)
        AVCheckSet[i] = g.vList.at(i).isActive;

    err = cudaMemcpy(d_AVCheckSet, AVCheckSet, g.vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //initVSet Init
    counterForIter = 0;
    for(auto iter = g.vList.at(0).value.begin(); iter != g.vList.at(0).value.end() && counterForIter < numOfInitV; iter++)
        initVSet[counterForIter++] = iter->first;

    err = cudaMemcpy(d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    for(int i = 0; i < g.vCount; i++)
    {
        counterForIter = 0;
        for(auto iter = g.vList.at(i).value.begin(); iter != g.vList.at(i).value.end(); iter++)
            vValueSet[i * numOfInitV + counterForIter++] = iter->second;
    }

    err = cudaMemcpy(d_vValueSet, vValueSet, g.vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);
    
    //Loop by groups of e
    for(auto eG : eSetG)
    {
        //Transfer data into GPU
        //info necessary: a group of e, values of v for initV, AVSet
        //eGroup.size() may be equal to the number of GPU threads
        //Size: eGroup.size() + vCount * numOfInitV + AVSet.size()
        
        //eSrcSet, eDstSet & eWeightSet
        if(eG.size() == 0) continue;

        for(int i = 0; i < eG.size(); i++)
        {
            eSrcSet[i] = eG.at(i).src;
            eDstSet[i] = eG.at(i).dst;
            eWeightSet[i] = eG.at(i).weight;
        }

        //msg info Set
        for(int i = 0; i < eG.size() * numOfInitV; i++)
        {
            mInitVSet[i] = NULLMSG;
            mDstSet[i] = NULLMSG;
            mValueSet[i] = NULLMSG;
        }

        //GPU kernel function (one GPU thread handles a edge)

        //Memory copy
        err = cudaMemcpy(d_eSrcSet, eSrcSet, eG.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_eDstSet, eDstSet, eG.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_eWeightSet, eWeightSet, eG.size() * sizeof(double), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mInitVSet, mInitVSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mDstSet, mDstSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mValueSet, mValueSet, eG.size() * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

        //Kernel Execution
        err = MSGGen_kernel_exec(eG.size(), d_AVCheckSet, d_eSrcSet, d_eDstSet, d_eWeightSet, numOfInitV, d_initVSet, d_vValueSet, d_mDstSet, d_mInitVSet, d_mValueSet);

        //Re-package the data
        //info necessary: mSetCheck table for each e in eGroup
        //Size: eGroup.size() * numOfInitV

        //Memory copyback
        err = cudaMemcpy(mDstSet, d_mDstSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(mInitVSet, d_mInitVSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(mValueSet, d_mValueSet, eG.size() * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //CPU gathers mSet from each mSetCheck table
        for(int i = 0; i < eG.size() * numOfInitV; i++)
        {
            if(mDstSet[i] != NULLMSG)
                mSet.insertMsg(Message(mInitVSet[i], mDstSet[i], mValueSet[i]));
        }
    }
}

void BellmanFordGPU::MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source)
{
    //Availability check
    if(source.mSet.size() == 0) return;

    //Organise m group
    int numG = source.mSet.size() / mPerMSGSet + 1;
    std::vector<MessageSet> mSetG = std::vector<MessageSet>();
    for(int i = 0; i < numG; i++) mSetG.push_back(MessageSet());
    for(int i = 0; i < source.mSet.size(); i++) mSetG.at(i / mPerMSGSet).insertMsg(source.mSet.at(i));

    //Init
    int counterForIter = 0;
    
    cudaError_t err = cudaSuccess;

    //initVSet Init
    counterForIter = 0;
    for(auto iter = g.vList.at(0).value.begin(); iter != g.vList.at(0).value.end() && counterForIter < numOfInitV; iter++)
    {
        initVSet[counterForIter] = iter->first;
        counterForIter++;
    }

    err = cudaMemcpy(d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //mMergedMSGValueSet Init
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = INVALID_MASSAGE;

    //Transform to the long long int form which CUDA can do atomic ops
    //unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt(mMergedMSGValueSet[i]);

    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    //Loop by groups of msg
    for(auto mS : mSetG)
    {
        //msg info Set
        for(int i = 0; i < mS.mSet.size(); i++)
        {
            mInitVSet[i] = mS.mSet.at(i).src;
            mDstSet[i] = mS.mSet.at(i).dst;
            mValueSet[i] = mS.mSet.at(i).value;
            mValueTSet[i] = doubleAsLongLongInt(mValueSet[i]);
        }

        //Memory copy
        err = cudaMemcpy(d_mInitVSet, mInitVSet, mS.mSet.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mDstSet, mDstSet, mS.mSet.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_mValueTSet, mValueTSet, mS.mSet.size() * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        
        //Kernel Execution
        err = MSGMerge_kernel_exec(d_mTransformedMergedMSGValueSet, numOfInitV, d_initVSet, mS.mSet.size(), d_mDstSet, d_mInitVSet, d_mValueTSet);
        
        cudaDeviceSynchronize();
    }

    //Re-package the data
    //Memory copyback
    err = cudaMemcpy(mTransformedMergedMSGValueSet, d_mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    //Transform back to original double form
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = longLongIntAsDouble(mTransformedMergedMSGValueSet[i]);

    //Package mMergedMSGValueSet to result mSet
    for(int i = 0; i < g.vCount * numOfInitV; i++)
    {
        if(mMergedMSGValueSet[i] != INVALID_MASSAGE)
        {
            int dst = i / numOfInitV;
            int initV = initVSet[i % numOfInitV];
            result.insertMsg(Message(initV, dst, mMergedMSGValueSet[i]));
        }
    }
}

void BellmanFordGPU::MSGGenMerge(const Graph &g, const std::set<int> &activeVertice, MessageSet &mSet)
{
    //Generate merged msgs directly

    //Availability check
    if(g.vCount == 0) return;

    //Organise e group
    int ePerEdgeSet = NUMOFGPUCORE;
    int numG = g.eCount / ePerEdgeSet + 1;
    auto eSetG = std::vector<std::vector<Edge>>();
    for(int i = 0; i < numG; i++) eSetG.push_back(std::vector<Edge>());

    //Filter of valid message through isActive in source vertex
    for(int i = 0, validEdgeNum = 0; i < g.eCount; i++)
    {
        if(g.vList.at(g.eList.at(i).src).isActive)
            eSetG.at(validEdgeNum++ / ePerEdgeSet).push_back(g.eList.at(i));
    }

    //Init
    int counterForIter = 0;

    //Memory allocation
    cudaError_t err = cudaSuccess;

    //AVCheckSet Init
    for(int i = 0; i < g.vCount; i++)
        AVCheckSet[i] = g.vList.at(i).isActive;

    err = cudaMemcpy(d_AVCheckSet, AVCheckSet, g.vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //initVSet Init
    counterForIter = 0;
    for(auto iter = g.vList.at(0).value.begin(); iter != g.vList.at(0).value.end() && counterForIter < numOfInitV; iter++)
        initVSet[counterForIter++] = iter->first;

    err = cudaMemcpy(d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    for(int i = 0; i < g.vCount; i++)
    {
        counterForIter = 0;
        for(auto iter = g.vList.at(i).value.begin(); iter != g.vList.at(i).value.end(); iter++)
            vValueSet[i * numOfInitV + counterForIter++] = iter->second;
    }

    err = cudaMemcpy(d_vValueSet, vValueSet, g.vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //mMergedMSGValueSet Init
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = INVALID_MASSAGE;

    //Transform to the long long int form which CUDA can do atomic ops
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt(mMergedMSGValueSet[i]);

    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    for(auto eG : eSetG)
    {
        //Transfer data into GPU
        //info necessary: a group of e, values of v for initV, AVSet
        //eGroup.size() may be equal to the number of GPU threads
        //Size: eGroup.size() + vCount * numOfInitV + AVSet.size()

        //eSrcSet, eDstSet & eWeightSet
        if(eG.size() == 0) continue;

        for(int i = 0; i < eG.size(); i++)
        {
            eSrcSet[i] = eG.at(i).src;
            eDstSet[i] = eG.at(i).dst;
            eWeightSet[i] = eG.at(i).weight;
        }

        err = cudaMemcpy(d_eSrcSet, eSrcSet, eG.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_eDstSet, eDstSet, eG.size() * sizeof(int), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_eWeightSet, eWeightSet, eG.size() * sizeof(double), cudaMemcpyHostToDevice);
        
        //Kernel Execution
        err = MSGGenMerge_kernel_exec(d_mTransformedMergedMSGValueSet, d_AVCheckSet, numOfInitV, d_initVSet, d_vValueSet, eG.size(), d_eSrcSet, d_eDstSet, d_eWeightSet);

        cudaDeviceSynchronize();
    }

    //Re-package the data
    //Memory copyback
    err = cudaMemcpy(mTransformedMergedMSGValueSet, d_mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    //Transform back to original double form
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = longLongIntAsDouble(mTransformedMergedMSGValueSet[i]);

    //Package mMergedMSGValueSet to result mSet
    for(int i = 0; i < g.vCount * numOfInitV; i++)
    {
        if(mMergedMSGValueSet[i] != INVALID_MASSAGE)
        {
            int dst = i / numOfInitV;
            int initV = initVSet[i % numOfInitV];
            mSet.insertMsg(Message(initV, dst, mMergedMSGValueSet[i]));
        }
    }
}

void BellmanFordGPU::MSGApply_array(int vCount, int numOfInitV, int *initVSet, bool *AVCheckSet, double *vValues, double *mValues)
{
    //Availability check
    if(vCount == 0) return;

    //CUDA init
    cudaError_t err = cudaSuccess;

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    err = cudaMemcpy(this->d_vValueSet, vValues, vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //AVCheckSet Init
    for(int i = 0; i < vCount; i++) AVCheckSet[i] = false;

    err = cudaMemcpy(this->d_AVCheckSet, AVCheckSet, vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //Apply msgs to v
    int mGCount = 0;
    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(mValues[i] != INVALID_MASSAGE) //Adding msgs to batchs
        {
            this->mInitVSet[mGCount] = initVSet[i % numOfInitV];
            this->mDstSet[mGCount] = i / numOfInitV;
            this->mValueSet[mGCount] = mValues[i];
            mGCount++;
        }
        if(mGCount == this->mPerMSGSet || i == vCount * numOfInitV - 1) //A batch of msgs will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_mInitVSet, this->mInitVSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, this->mValueSet, mGCount * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGApply_kernel_exec(numOfInitV, this->d_initVSet, this->d_vValueSet, mGCount, this->d_mDstSet, this->d_mInitVSet, this->d_mValueSet, this->d_AVCheckSet);

            mGCount = 0;
        }
    }

    //Re-package the data

    //Memory copyback
    err = cudaMemcpy(AVCheckSet, this->d_AVCheckSet, vCount * sizeof(bool), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(vValues, this->d_vValueSet, vCount * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);
}

void BellmanFordGPU::MSGGen_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, int &numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, bool *AVCheckSet)
{
    //Availability check
    if(vCount == 0) return;

    //CUDA init
    cudaError_t err = cudaSuccess;

    //AVCheckSet Init
    err = cudaMemcpy(this->d_AVCheckSet, AVCheckSet, vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    err = cudaMemcpy(this->d_vValueSet, vValues, vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //Gen msgs using e
    numOfMSG = 0;
    int eGCount = 0;
    for(int i = 0; i < eCount; i++)
    {
        if(AVCheckSet[eSrcSet[i]]) //Add es to batchs
        {
            this->eSrcSet[eGCount] = eSrcSet[i];
            this->eDstSet[eGCount] = eDstSet[i];
            this->eWeightSet[eGCount] = eWeightSet[i];
            this->mInitVSet[eGCount] = NULLMSG;
            this->mDstSet[eGCount] = NULLMSG;
            this->mValueSet[eGCount] = NULLMSG;
            eGCount++;
        }
        if(eGCount == this->ePerEdgeSet || i == eCount - 1) //A batch of es will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_eSrcSet, this->eSrcSet, eGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_eDstSet, this->eDstSet, eGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_eWeightSet, this->eWeightSet, eGCount * sizeof(double), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mInitVSet, this->mInitVSet, eGCount * numOfInitV * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, eGCount * numOfInitV * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, this->mValueSet, eGCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGGen_kernel_exec(eGCount, this->d_AVCheckSet, this->d_eSrcSet, this->d_eDstSet, this->d_eWeightSet, numOfInitV, this->d_initVSet, this->d_vValueSet, this->d_mDstSet, this->d_mInitVSet, this->d_mValueSet);

            //Re-package the data
            //info necessary: mSetCheck table for each e in eGroup
            //Size: eGroup.size() * numOfInitV

            //Memory copyback
            err = cudaMemcpy(this->mDstSet, this->d_mDstSet, eGCount * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
            err = cudaMemcpy(this->mInitVSet, this->d_mInitVSet, eGCount * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
            err = cudaMemcpy(this->mValueSet, this->d_mValueSet, eGCount * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);

            //Gathers mSet from each mSetCheck table
            for(int i = 0; i < eGCount * numOfInitV; i++)
            {
                if(this->mDstSet[i] != NULLMSG)
                {
                    mInitVSet[numOfMSG] = this->mInitVSet[i];
                    mDstSet[numOfMSG] = this->mDstSet[i];
                    mValueSet[numOfMSG] = this->mValueSet[i];
                    numOfMSG++;
                }
            }

            eGCount = 0;
        }
    }
}

void BellmanFordGPU::MSGMerge_array(int vCount, int numOfInitV, int *initVSet, int numOfMSG, int *mInitVSet, int *mDstSet, double *mValueSet, double *mValues)
{
    //Availability check
    if(numOfMSG == 0) return;

    //Init
    cudaError_t err = cudaSuccess;

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //mMergedMSGValueSet Init
    for(int i = 0; i < vCount * numOfInitV; i++)
        mValues[i] = INVALID_MASSAGE;

    //Transform to the long long int form which CUDA can do atomic ops
    //unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for(int i = 0; i < vCount * numOfInitV; i++)
        this->mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt(mValues[i]);
    
    err = cudaMemcpy(this->d_mTransformedMergedMSGValueSet, this->mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    //Batch merge msgs
    int mGCount = 0;
    for(int i = 0; i < numOfMSG; i++)
    {
        //msg info Set
        this->mInitVSet[mGCount] = mInitVSet[i];
        this->mDstSet[mGCount] = mDstSet[i];
        this->mValueSet[mGCount] = mValueSet[i];
        this->mValueTSet[mGCount] = doubleAsLongLongInt(this->mValueSet[mGCount]);
        if(mGCount == this->mPerMSGSet || i == numOfMSG - 1) //A batch of msgs will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_mInitVSet, this->mInitVSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueTSet, this->mValueTSet, mGCount * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGMerge_kernel_exec(this->d_mTransformedMergedMSGValueSet, numOfInitV, this->d_initVSet, mGCount, this->d_mDstSet, this->d_mInitVSet, this->d_mValueTSet);

            mGCount = 0;
        }
    }

    //Re-package the data
    //Memory copyback
    err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

    //Transform back to original double form
    for(int i = 0; i < vCount * numOfInitV; i++)
        mValues[i] = longLongIntAsDouble(mTransformedMergedMSGValueSet[i]);
}

void BellmanFordGPU::MSGGenMerge_array(int vCount, int eCount, int numOfInitV, int *initVSet, double *vValues, int *eSrcSet, int *eDstSet, double *eWeightSet, double *mValues, bool *AVCheckSet)
{
    //Generate merged msgs directly

    //Availability check
    if(vCount == 0) return;

    //Memory allocation
    cudaError_t err = cudaSuccess;

    //AVCheckSet Init
    err = cudaMemcpy(this->d_AVCheckSet, AVCheckSet, g.vCount * sizeof(bool), cudaMemcpyHostToDevice);

    //initVSet Init
    err = cudaMemcpy(this->d_initVSet, initVSet, numOfInitV * sizeof(int), cudaMemcpyHostToDevice);

    //vValueSet Init
    err = cudaMemcpy(this->d_vValueSet, vValues, g.vCount * numOfInitV * sizeof(double), cudaMemcpyHostToDevice);

    //mMergedMSGValueSet Init
    for(int i = 0; i < vCount * numOfInitV; i++)
        mValues[i] = INVALID_MASSAGE;

    //Transform to the long long int form which CUDA can do atomic ops
    //unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for(int i = 0; i < vCount * numOfInitV; i++)
        this->mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt(mValues[i]);
    
    err = cudaMemcpy(this->d_mTransformedMergedMSGValueSet, this->mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    //e batch processing
    int eGCount = 0;
    for(int i = 0; i < eCount; i++)
    {
        if(AVCheckSet[eSrcSet[i]]) //Add es to batchs
        {
            this->eSrcSet[eGCount] = eSrcSet[i];
            this->eDstSet[eGCount] = eDstSet[i];
            this->eWeightSet[eGCount] = eWeightSet[i];
            eGCount++;
        }
        if(eGCount == this->ePerEdgeSet || i == eCount - 1) //A batch of es will be transferred into GPU. Don't forget last batch!
        {
            //Memory copy
            err = cudaMemcpy(this->d_eSrcSet, this->eSrcSet, eGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_eDstSet, this->eDstSet, eGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_eWeightSet, this->eWeightSet, eGCount * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            err = MSGGenMerge_kernel_exec(this->d_mTransformedMergedMSGValueSet, this->d_AVCheckSet, numOfInitV, this->d_initVSet, this->d_vValueSet, eGCount, this->d_eSrcSet, this->d_eDstSet, this->d_eWeightSet);

            eGCount = 0;
        }
    }

    //Re-package the data
    //Memory copyback
    err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet, vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    
    //Transform back to original double form
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mValues[i] = longLongIntAsDouble(mTransformedMergedMSGValueSet[i]);
}
