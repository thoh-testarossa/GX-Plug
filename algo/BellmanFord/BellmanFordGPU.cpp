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
    this->mTransformedMergedMSGValueSet = new unsigned long long int [vertexLimit * numOfInitV];
    err = cudaMalloc((void **)&d_mTransformedMergedMSGValueSet, numOfInitV * vertexLimit * sizeof(unsigned long long int));

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
        {
            activeVertices.insert(i);
            //Test
            /*
            std::cout << i << ": ";
            for(int j = 0; j < this->numOfInitV; j++)
                std::cout << g.verticesValue.at(i * this->numOfInitV + j) << " ";
            std::cout << std::endl;
            */
        }
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
            //Test
            //if(dst == 7)
            //    std::cout << this->mMergedMSGValueSet[i] << "///////" << std::endl;
        }
    }

    //Test
    //std::cout << mSet.mSet.size() << std::endl;
    //for(const auto &m : mSet.mSet) std::cout << m.src << "->" << m.dst << ": " << m.value << std::endl;
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

                //Test
                /*
                std::cout << mGSet.mSet.size() << " -> " << r_mGSet.mSet.size() << std::endl;
                for(int j = 0; j < mGSet.mSet.size() && j < r_mGSet.mSet.size(); j++)
                {
                    std::cout << "(" << mGSet.mSet.at(j).src << " -> " << mGSet.mSet.at(j).dst << ": " << mGSet.mSet.at(j).value << ")";
                    std::cout << " - (" << reflectIndex.at(reversedIndex.at(mGSet.mSet.at(j).dst)) << " -> " << reversedIndex.at(mGSet.mSet.at(j).dst) << ") - ";
                    std::cout << "(" << r_mGSet.mSet.at(j).src << " -> " << r_mGSet.mSet.at(j).dst << ": " << mGSet.mSet.at(j).value << ")" << std::endl;
                }
                */

                for(int j = 0; j < r_mGSet.mSet.size(); j++)
                {
                    this->mInitVIndexSet[j] = vSet[r_mGSet.mSet.at(j).src].initVIndex;
                    this->mDstSet[j] = r_mGSet.mSet.at(j).dst;
                    this->mValueSet[j] = (double)r_mGSet.mSet.at(j).value;
                }

                //v reflection
                r_vSet.clear();
                for(int j = 0; j < reflectIndex.size(); j++)
                    r_vSet.emplace_back(j, false, vSet[reflectIndex.at(j)].initVIndex);

                //vValue reflection
                r_vValueSet.clear();
                r_vValueSet.reserve(mPerMSGSet * numOfInitV);
                r_vValueSet.assign(mPerMSGSet * numOfInitV, INT32_MAX >> 1);
                for(int j = 0; j < reflectIndex.size(); j++)
                {
                    for(int k = 0; k < numOfInitV; k++)
                        r_vValueSet.at(j * numOfInitV + k) = vValues[reflectIndex[j] * numOfInitV + k];
                }

                //vSet & vValueSet Init
                err = MSGApply_GPU_VVCopy(d_vSet, &r_vSet[0],
                                    d_vValueSet, (double *)&r_vValueSet[0],
                                    reflectIndex.size(), numOfInitV);
            }
            else
            {
                //Use original msg
                for(int j = 0; j < mGSet.mSet.size(); j++)
                {
                    this->mInitVIndexSet[j] = vSet[mGSet.mSet.at(j).src].initVIndex;
                    this->mDstSet[j] = mGSet.mSet.at(j).dst;
                    this->mValueSet[j] = (double)mGSet.mSet.at(j).value;
                }
            }

            //MSG memory copy
            err = cudaMemcpy(this->d_mInitVIndexSet, this->mInitVIndexSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, this->mValueSet, mGCount * sizeof(double), cudaMemcpyHostToDevice);

            //Kernel Execution
            for(int j = 0; j < mGCount; j += NUMOFGPUCORE)
            {
                int msgNumUsedForExec = (mGCount - j > NUMOFGPUCORE) ? NUMOFGPUCORE : (mGCount - j);

                err = MSGApply_kernel_exec(this->d_vSet, numOfInitV, this->d_initVSet, this->d_vValueSet, msgNumUsedForExec,
                                           &this->d_mDstSet[j], &this->d_mInitVIndexSet[j], &this->d_mValueSet[j]);
            }

            //Deflection
            if(needReflect)
            {
                err = cudaMemcpy(&r_vSet[0], this->d_vSet, reflectIndex.size() * sizeof(Vertex), cudaMemcpyDeviceToHost);
                err = cudaMemcpy((double *)&r_vValueSet[0], this->d_vValueSet, reflectIndex.size() * numOfInitV * sizeof(double),
                                 cudaMemcpyDeviceToHost);

                for(int j = 0; j < reflectIndex.size(); j++)
                {
                    vSet[reflectIndex[j]] = r_vSet[j];
                    //Don't forget to deflect vertexID in Vertex obj!!
                    vSet[reflectIndex[j]].vertexID = reflectIndex[j];
                    for(int k = 0; k < numOfInitV; k++)
                        vValues[reflectIndex[j] * numOfInitV + k] = r_vValueSet[j * numOfInitV + k];
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

    //Invalid message init
    for(int i = 0; i < vCount * numOfInitV; i++) mValues[i] = (VertexValueType)INVALID_MASSAGE;

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

    //Init for possible reflection
    //Maybe can use lambda style?
    bool *tmp_AVCheckList = new bool [vCount];
    auto tmp_o_g = Graph<VertexValueType>(0);
    if(needReflect)
    {
        for(int i = 0; i < vCount; i++) tmp_AVCheckList[i] = vSet[i].isActive;
        tmp_o_g = Graph<VertexValueType>(vCount, 0, numOfInitV, initVSet, nullptr, nullptr, nullptr, tmp_AVCheckList);
        tmp_o_g.verticesValue.reserve(vCount * numOfInitV);
        tmp_o_g.verticesValue.insert(tmp_o_g.verticesValue.begin(), vValues, vValues + (numOfInitV * vCount));
    }
    //This checkpoint is to used to prevent from mistaking mValues gathering in deflection
    bool *isDst = new bool [vCount];
    for(int i = 0; i < vCount; i++) isDst[i] = false;

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
                    //Only dst receives message
                    isDst[eSet[i].dst] = true;

                    //Test
                    /*
                    if(eSet[i].dst == 7)
                    {
                        std::cout << "**" << std::endl;
                        std::cout << i << " " << eGCount << ": " << eSet[i].src << " -> " << eSet[i].dst << ": "
                                  << eSet[i].weight << std::endl;
                        for(int k = 0; k < numOfInitV; k++)
                            std::cout << vValues[eSet[i].src * numOfInitV + k] << " ";
                        std::cout << std::endl;
                        for(int k = 0; k < numOfInitV; k++)
                            std::cout << vValues[eSet[i].dst * numOfInitV + k] << " ";
                        std::cout << std::endl;
                        std::cout << "**" << std::endl;
                    }
                    */
                    break;
                }
            }
        }
        if(eGCount == this->ePerEdgeSet || i == eCount - 1) //A batch of es will be transferred into GPU. Don't forget last batch!
        {
            //Test
            //std::cout << "*" << eGCount << std::endl;

            auto reflectIndex = std::vector<int>();
            auto reversedIndex = std::vector<int>();

            auto r_g = Graph<VertexValueType>(0);

            //Reflection
            if(needReflect)
            {
                //bool *tmp_AVCheckList = new bool [vCount];
                //for(int i = 0; i < vCount; i++) tmp_AVCheckList[i] = vSet[i].isActive;

                //auto tmp_o_g = Graph<VertexValueType>(vCount, 0, numOfInitV, initVSet, nullptr, nullptr, nullptr, tmp_AVCheckList);
                //tmp_o_g.verticesValue.reserve(vCount * numOfInitV);
                //tmp_o_g.verticesValue.insert(tmp_o_g.verticesValue.begin(), vValues, vValues + (numOfInitV * vCount));

                //Test
                //std::cout << tmp_o_g.vCount << " " << tmp_o_g.vList.size() << " " << tmp_o_g.verticesValue.size() << std::endl;
                //std::cout << eGSet.size() << " " << eGCount << std::endl;

                r_g = this->reflectG(tmp_o_g, eGSet, reflectIndex, reversedIndex);

                //Test
                //std::cout << "***" << reversedIndex[7] << std::endl;
                //std::cout << r_g.vCount << " " << r_g.vList.size() << " " << r_g.verticesValue.size() << std::endl;
                //for(int i = 0; i < r_g.vCount * numOfInitV; i++)
                //    std::cout << r_g.verticesValue.at(i) << " ";
                //std::cout << std::endl << "**********************************************************" << std::endl;
                //for(const auto &e : eGSet) std::cout << e.src << "->" << e.dst << ": " << e.weight << std::endl;
                //for(const auto &ri : reflectIndex) std::cout << ri << " ";
                //std::cout << std::endl;

                err = MSGGenMerge_GPU_MVCopy(this->d_vSet, &r_g.vList[0],
                                             this->d_vValueSet, (double *)&r_g.verticesValue[0],
                                             this->d_mTransformedMergedMSGValueSet,
                                             this->mTransformedMergedMSGValueSet,
                                             r_g.vCount, numOfInitV);

                //Test
                //for(int j = 0; j < r_g.vCount * numOfInitV; j++) std::cout << this->mTransformedMergedMSGValueSet[j] << " ";
                //std::cout << std::endl;

                err = cudaMemcpy(this->d_eGSet, &r_g.eList[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);
            }
            else
                err = cudaMemcpy(this->d_eGSet, &eGSet[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);

            //Kernel Execution (no matter whether g is reflected or not)
            for(int j = 0; j < eGCount; j += NUMOFGPUCORE)
            {
                int edgeNumUsedForExec = (eGCount - j > NUMOFGPUCORE) ? NUMOFGPUCORE : (eGCount - j);

                err = MSGGenMerge_kernel_exec(this->d_mTransformedMergedMSGValueSet, this->d_vSet, numOfInitV,
                                              this->d_initVSet, this->d_vValueSet, edgeNumUsedForExec, &this->d_eGSet[j]);
            }

            //Deflection
            if(needReflect)
            {
                //Re-package the data
                //Memory copy back
                err = cudaMemcpy(this->mTransformedMergedMSGValueSet, this->d_mTransformedMergedMSGValueSet,
                                 r_g.vCount * numOfInitV * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

                //Test
                //for(int j = 0; j < r_g.vCount * numOfInitV; j++) std::cout << this->mTransformedMergedMSGValueSet[j] << " ";
                //std::cout << std::endl;

                //Valid message transformed back to original double form (deflection)
                for (int j = 0; j < r_g.vCount * numOfInitV; j++)
                {
                    int o_dst = reflectIndex[j / numOfInitV];
                    //If the v the current msg point to is not a dst, it should not be copied back because the current msg value is not correct)
                    if(isDst[o_dst])
                    {
                        if(mValues[o_dst * numOfInitV + j % numOfInitV] > (VertexValueType) (longLongIntAsDouble(this->mTransformedMergedMSGValueSet[j])))
                            mValues[o_dst * numOfInitV + j % numOfInitV] = (VertexValueType) (longLongIntAsDouble(
                                this->mTransformedMergedMSGValueSet[j]));
                    }
                }

                //Test
                //for(int j = 0; j < reflectIndex.size(); j++)
                //{
                //    for(int k = 0; k < numOfInitV; k++)
                //        std::cout << mValues[reflectIndex.at(j) * numOfInitV + k] << " ";
                //    std::cout << std::endl;
                //}
                //for(int j = 0; j < numOfInitV; j++)
                //    std::cout << "**" << mValues[7 * numOfInitV + j] << " ";
                //std::cout << std::endl;
                //std::cout << isDst[7] << std::endl;
            }
            else;

            //Checkpoint reset
            eGCount = 0;
            eGSet.clear();
            for(int j = 0; j < vCount; j++) isDst[j] = false;
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
