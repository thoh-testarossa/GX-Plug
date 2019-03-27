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

//Needed to be moved to Init()
BellmanFordGPU::BellmanFordGPU()
{
}

void BellmanFordGPU::Init(Graph &g, std::set<int> &activeVertice, const std::vector<int> &initVList)
{
    BellmanFord::Init(g, activeVertice, initVList);
}

void BellmanFordGPU::Deploy(Graph &g, int numOfInitV)
{
    BellmanFord::Deploy(g, numOfInitV);

    cudaError_t err = cudaSuccess;

    this->mPerMSGSet = NUMOFGPUCORE;
    this->ePerEdgeSet = NUMOFGPUCORE;

    this->initVSet = new int [numOfInitV];
    err = cudaMalloc((void **)&this->d_initVSet, this->numOfInitV * sizeof(int));

    this->vValueSet = new double [g.vCount * this->numOfInitV];
    err = cudaMalloc((void **)&this->d_vValueSet, g.vCount * this->numOfInitV * sizeof(double));

    this->AVCheckSet = new bool [g.vCount];
    err = cudaMalloc((void **)&this->d_AVCheckSet, g.vCount * sizeof(bool));

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

    this->activeVerticeSet = new int [g.vCount];
    err = cudaMalloc((void **)&this->d_activeVerticeSet, g.vCount * sizeof(int));
}

void BellmanFordGPU::Free(Graph &g)
{
    BellmanFord::Free(g);

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
}

void BellmanFordGPU::MSGApply(Graph &g, std::set<int> &activeVertice, const MessageSet &mSet)
{
    //Availability check
    if(g.vCount == 0) return;

    //Organise m group
    //int mPerMSGSet = NUMOFGPUCORE;
    int numG = mSet.mSet.size() / mPerMSGSet + 1;
    std::vector<MessageSet> mSetG = std::vector<MessageSet>();
    for(int i = 0; i < numG; i++) mSetG.push_back(MessageSet());
    for(int i = 0; i < mSet.mSet.size(); i++) mSetG.at(i / mPerMSGSet).insertMsg(mSet.mSet.at(i));

    //Init
    //const int numOfInitV = g.vList.at(0).value.size();

    int counterForIter = 0;

    //int *initVSet = new int [numOfInitV];
    //double *vValueSet = new double [g.vCount * numOfInitV];

    //int *mDstSet = new int [mPerMSGSet];
    //int *mInitVSet = new int [mPerMSGSet];
    //double *mValueSet = new double [mPerMSGSet];
    //bool *AVCheckSet = new bool [g.vCount];

    //CUDA init
    //int *d_initVSet = nullptr, *d_mDstSet = nullptr, *d_mInitVSet = nullptr;
    //double *d_vValueSet = nullptr, *d_mValueSet = nullptr;
    //bool *d_AVCheckSet = nullptr;

    //Memory allocation
    cudaError_t err = cudaSuccess;
    //err = cudaMalloc((void **)&d_initVSet, numOfInitV * sizeof(int));
    //err = cudaMalloc((void **)&d_mInitVSet, mPerMSGSet * sizeof(int));
    //err = cudaMalloc((void **)&d_mDstSet, mPerMSGSet * sizeof(int));
    //err = cudaMalloc((void **)&d_vValueSet, g.vCount * numOfInitV * sizeof(double));
    //err = cudaMalloc((void **)&d_mValueSet, mPerMSGSet * sizeof(double));
    //err = cudaMalloc((void **)&d_AVCheckSet, g.vCount * sizeof(bool));
    
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

        //numOfInitV

        //initVSet

        //vValueSet

        //msg info Set
        for(int i = 0; i < mS.mSet.size(); i++)
        {
            mInitVSet[i] = mS.mSet.at(i).src;
            mDstSet[i] = mS.mSet.at(i).dst;
            mValueSet[i] = mS.mSet.at(i).value;
        }

        //Test
        /*
        for(int i = 0; i < mS.mSet.size(); i++)
            std::cout << mInitVSet[i] << " " << mDstSet[i] << " " << mValueSet[i] << std::endl;
        std::cout << std::endl;
        */
        //Test end

        //AVCheckSet

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

    //Free CUDA mem
    //err = cudaFree(d_initVSet);
    //err = cudaFree(d_vValueSet);
    //err = cudaFree(d_mDstSet);
    //err = cudaFree(d_mInitVSet);
    //err = cudaFree(d_mValueSet);
    //err = cudaFree(d_AVCheckSet);
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
    //int validEdgeNum = 0;
    for(int i = 0; i < g.eCount; i++){
        //if(g.vList.at(g.eList.at(i).src).isActive){
            //eSetG.at(validEdgeNum++ / ePerEdgeSet).push_back(g.eList.at(i));
        eSetG.at(i / ePerEdgeSet).push_back(g.eList.at(i));
        //}
    }

    //Init
    //const int numOfInitV = g.vList.at(0).value.size();
    const int numOfAV = activeVertice.size();

    int counterForIter = 0;

    //int *eSrcSet = new int [ePerEdgeSet];
    //int *eDstSet = new int [ePerEdgeSet];
    //double *eWeightSet = new double [ePerEdgeSet];
    
    //int *activeVerticeSet = new int [numOfAV];

    //int *initVSet = new int [numOfInitV];
    //double *vValueSet = new double [g.vCount * numOfInitV];

    //int *mDstSet = new int [numOfInitV * ePerEdgeSet];
    //int *mInitVSet = new int [numOfInitV * ePerEdgeSet];
    //double *mValueSet = new double [numOfInitV * ePerEdgeSet];

    //CUDA init
    //int *d_eSrcSet = nullptr, *d_eDstSet = nullptr;
    //double *d_eWeightSet = nullptr;
    //int *d_initVSet = nullptr;
    //double *d_vValueSet = nullptr;
    //int *d_activeVerticeSet = nullptr;
    //int *d_mDstSet = nullptr, *d_mInitVSet = nullptr;
    //double *d_mValueSet = nullptr;

    //Memory allocation
    cudaError_t err = cudaSuccess;
    //err = cudaMalloc((void **)&d_eSrcSet, ePerEdgeSet * sizeof(int));
    //err = cudaMalloc((void **)&d_eDstSet, ePerEdgeSet * sizeof(int));
    //err = cudaMalloc((void **)&d_eWeightSet, ePerEdgeSet * sizeof(double));
    //err = cudaMalloc((void **)&d_initVSet, numOfInitV * sizeof(int));
    //err = cudaMalloc((void **)&d_vValueSet, g.vCount * numOfInitV * sizeof(double));
    //err = cudaMalloc((void **)&d_activeVerticeSet, numOfAV * sizeof(int));
    //err = cudaMalloc((void **)&d_mDstSet, numOfInitV * g.eCount * sizeof(int));
    //err = cudaMalloc((void **)&d_mInitVSet, numOfInitV * g.eCount * sizeof(int));
    //err = cudaMalloc((void **)&d_mValueSet, numOfInitV * g.eCount * sizeof(double));

    //activeVerticeSet Init
    counterForIter = 0;
    for(auto iter = activeVertice.begin(); iter != activeVertice.end(); iter++)
        activeVerticeSet[counterForIter++] = *iter;

    err = cudaMemcpy(d_activeVerticeSet, activeVerticeSet, numOfAV * sizeof(int), cudaMemcpyHostToDevice);

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

    //Test
    /*
    for(int i = 0; i < g.vCount * numOfInitV; i++)
    {
            std::cout << vValueSet[i] << " ";
            if(i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    //Test end
    
    //Loop by groups of e
    for(auto eG : eSetG)
    {
        //Transfer data into GPU
        //info necessary: a group of e, values of v for initV, AVSet
        //eGroup.size() may be equal to the number of GPU threads
        //Size: eGroup.size() + vCount * numOfInitV + AVSet.size()
        
        //activeVerticeSet

        //eSrcSet, eDstSet & eWeightSet

        //if(eG.size() == 0)continue;

        for(int i = 0; i < eG.size(); i++)
        {
            eSrcSet[i] = eG.at(i).src;
            eDstSet[i] = eG.at(i).dst;
            eWeightSet[i] = eG.at(i).weight;
        }

        //numOfInitV

        //initVSet

        //vValueSet

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
        err = MSGGen_kernel_exec(eG.size(), numOfAV, d_activeVerticeSet, d_AVCheckSet, d_eSrcSet, d_eDstSet, d_eWeightSet, numOfInitV, d_initVSet, d_vValueSet, d_mDstSet, d_mInitVSet, d_mValueSet);

        //Re-package the data
        //info necessary: mSetCheck table for each e in eGroup
        //Size: eGroup.size() * numOfInitV

        //Memory copyback
        err = cudaMemcpy(mDstSet, d_mDstSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(mInitVSet, d_mInitVSet, eG.size() * numOfInitV * sizeof(int), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(mValueSet, d_mValueSet, eG.size() * numOfInitV * sizeof(double), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //Test
        /*
        for(int i = 0; i < eG.size() * numOfInitV; i++)
        {
            if(mDstSet[i] != NULLMSG)
            {
                int eID = i / numOfInitV;
                std::cout << eSrcSet[eID] << " " << eDstSet[eID] << " " << eWeightSet[eID] 
                << ": " 
                << mInitVSet[i] << " " << mDstSet[i] << " " << mValueSet[i] << std::endl;
            }
        }
        //std::cout << "***************************" << std::endl;
        */
        //Test end
        
        //CPU gathers mSet from each mSetCheck table
        for(int i = 0; i < eG.size() * numOfInitV; i++)
        {
            if(mDstSet[i] != NULLMSG)
                mSet.insertMsg(Message(mInitVSet[i], mDstSet[i], mValueSet[i]));
        }
    }
    //Free CUDA mem
    //err = cudaFree(d_activeVerticeSet);
    //err = cudaFree(d_eSrcSet);
    //err = cudaFree(d_eDstSet);
    //err = cudaFree(d_eWeightSet);
    //err = cudaFree(d_initVSet);
    //err = cudaFree(d_vValueSet);
    //err = cudaFree(d_mDstSet);
    //err = cudaFree(d_mInitVSet);
    //err = cudaFree(d_mValueSet);
}

void BellmanFordGPU::MSGMerge(const Graph &g, MessageSet &result, const MessageSet &source)
{
    //Availability check
    if(source.mSet.size() == 0) return;

    //Organise m group
    //int mPerMSGSet = NUMOFGPUCORE;
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
    //Test
    //std::cout << cudaGetErrorString(err) << std::endl;
    //Test end

    //mMergedMSGValueSet Init
    double *mMergedMSGValueSet = new double [g.vCount * numOfInitV];
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = INT32_MAX;

    //Transform to the long long int form which CUDA can do atomic ops
    unsigned long long int *mTransformedMergedMSGValueSet = new unsigned long long int [g.vCount * numOfInitV];
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mTransformedMergedMSGValueSet[i] = doubleAsLongLongInt(mMergedMSGValueSet[i]);

    unsigned long long int *d_mTransformedMergedMSGValueSet = nullptr;
    err = cudaMalloc((void **)&d_mTransformedMergedMSGValueSet, numOfInitV * g.vCount * sizeof(unsigned long long int));
    //Test
    //std::cout << cudaGetErrorString(err) << std::endl;
    //Test end
    err = cudaMemcpy(d_mTransformedMergedMSGValueSet, mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    //Test
    //std::cout << cudaGetErrorString(err) << std::endl;
    //Test end

    //Transform to the long long int form which CUDA can do atomic ops
    unsigned long long int *mValueTSet = new unsigned long long int [mPerMSGSet];

    unsigned long long int *d_mValueTSet = nullptr;
    err = cudaMalloc((void **)&d_mValueTSet, mPerMSGSet * sizeof(unsigned long long int));
    //Test
    //std::cout << cudaGetErrorString(err) << std::endl;
    //Test end

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
        //Test
        //std::cout << cudaGetErrorString(err) << std::endl;
        //Test end
        err = cudaMemcpy(d_mDstSet, mDstSet, mS.mSet.size() * sizeof(int), cudaMemcpyHostToDevice);
        //Test
        //std::cout << cudaGetErrorString(err) << std::endl;
        //Test end
        err = cudaMemcpy(d_mValueTSet, mValueTSet, mS.mSet.size() * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        //Test
        //std::cout << cudaGetErrorString(err) << std::endl;
        //Test end

        //Kernel Execution
        err = MSGMerge_kernel_exec(d_mTransformedMergedMSGValueSet, numOfInitV, d_initVSet, mS.mSet.size(), d_mDstSet, d_mInitVSet, d_mValueTSet);
        //Test
        //std::cout << "a " << cudaGetErrorString(err) << std::endl;
        //Test end

        cudaDeviceSynchronize();
    }

    //Re-package the data
    //Memory copyback
    err = cudaMemcpy(mTransformedMergedMSGValueSet, d_mTransformedMergedMSGValueSet, g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    //Test
    //std::cout << cudaGetErrorString(err) << std::endl;
    //Test end

    //Transform back to original double form
    for(int i = 0; i < g.vCount * numOfInitV; i++)
        mMergedMSGValueSet[i] = longLongIntAsDouble(mTransformedMergedMSGValueSet[i]);

    //Package mMergedMSGValueSet to result mSet
    for(int i = 0; i < g.vCount * numOfInitV; i++)
    {
        if(mMergedMSGValueSet[i] != INT32_MAX)
        {
            int dst = i / numOfInitV;
            int initV = initVSet[i % numOfInitV];
            result.insertMsg(Message(initV, dst, mMergedMSGValueSet[i]));
        }
    }

    //Memory free
    cudaFree(d_mTransformedMergedMSGValueSet);
    cudaFree(d_mValueTSet);
}
