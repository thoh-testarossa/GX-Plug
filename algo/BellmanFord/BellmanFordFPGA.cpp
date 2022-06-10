#include "BellmanFordFPGA.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

template<typename VertexValueType, typename MessageValueType>
BellmanFordFPGA<VertexValueType, MessageValueType>::BellmanFordFPGA()
{
    InitFPGAEnv();
    std::cout << "FPGA env has set" << std::endl;
}

template<typename VertexValueType, typename MessageValueType>
void
BellmanFordFPGA<VertexValueType, MessageValueType>::InitFPGAEnv ()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Device device;
    bool found_device = false;
    cl_int err;

    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            for (size_t j = 0 ; j < devices.size() ; j++){
                device = devices[j];
                std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
                std::cout << deviceName << std::endl;
            }
        }
    }
    
    OCL_CHECK(err, cl::Context context_tmp(device, NULL, NULL, NULL, &err));
    context = context_tmp;
    OCL_CHECK(err, cl::CommandQueue q1(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, cl::CommandQueue q2(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, cl::CommandQueue q3(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, cl::CommandQueue q4(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
    
    queues.push_back(q1);
    queues.push_back(q2);
    queues.push_back(q3);
    queues.push_back(q4);
    std::string binaryFile = xcl::find_binary_file(device_name, "gs_top");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    OCL_CHECK(err, cl::Program program_tmp(context, devices, bins, NULL, &err));
    program = program_tmp;
    int index = -1;
    OCL_CHECK(err, cl::Kernel kernel1(program, "gs_top1", &err));
    krnls.push_back(kernel1);
    OCL_CHECK(err, cl::Kernel kernel2(program, "gs_top2", &err));
    krnls.push_back(kernel2);
    OCL_CHECK(err, cl::Kernel kernel3(program, "gs_top3", &err));
    krnls.push_back(kernel3);
    OCL_CHECK(err, cl::Kernel kernel4(program, "gs_top4", &err));
    krnls.push_back(kernel4);
}

template<typename VertexValueType, typename MessageValueType>
int
BellmanFordFPGA<VertexValueType, MessageValueType>::MSGGenMerge_array(
    int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,MessageValueType *mValues)
{
    cl_mem mem;
    std::vector<std::vector<cl_mem>> memTable(4,std::vector<cl_mem>(5)); 
    std::vector<cl::Event> eventList(4);
    int LoopNum = (512*4 < computeUnitCount) ? 4 : (computeUnitCount-1) / 512 + 1;
    int arrayCount = (((computeUnitCount-1) / (4 * 512)) + 1) * 512;
    // std::vector<int,aligned_allocator<int>> srcValue(arrayCount);
    // std::vector<int,aligned_allocator<int>> ewValue(arrayCount);
    // std::vector<int,aligned_allocator<int>> dstArray(arrayCount);
    // std::vector<int,aligned_allocator<int>> dstValue(arrayCount,0);
    // std::vector<int,aligned_allocator<int>> activeNode(arrayCount,0);

    std::vector<std::vector<int,aligned_allocator<int>>> srcValueTabel(LoopNum,std::vector<int,aligned_allocator<int>>(arrayCount));
    std::vector<std::vector<int,aligned_allocator<int>>> ewValueTabel(LoopNum,std::vector<int,aligned_allocator<int>>(arrayCount));
    std::vector<std::vector<int,aligned_allocator<int>>> dstArrayTabel(LoopNum,std::vector<int,aligned_allocator<int>>(arrayCount));
    std::vector<std::vector<int,aligned_allocator<int>>> dstValueTabel(LoopNum,std::vector<int,aligned_allocator<int>>(arrayCount,0));
    std::vector<std::vector<int,aligned_allocator<int>>> activeNodeTabel(LoopNum,std::vector<int,aligned_allocator<int>>(arrayCount,0));

    std::vector<int> bound;
    int oneLoopNum = computeUnitCount / LoopNum;
    for(int i = 0 ; i < LoopNum-1 ; i++)
    {
        bound.push_back(oneLoopNum);
        //std::cout <<" " <<bound[i] ;
    }
    bound.push_back(computeUnitCount - oneLoopNum*(LoopNum-1));
    //std::cout << bound[LoopNum-1] << std::endl;
    cl_int err;
    int CUindex = 0;
    for(int loop = 0 ; loop < LoopNum ; loop++)
    {   
        for(int i = 0 ; i < bound[loop] ; i++)
        {
            auto& cu = computeUnits[CUindex++];
            cu.destVertex.isActive = false;
            cu.srcVertex.isActive = false;
            srcValueTabel[loop][i] = cu.srcValue;
            ewValueTabel[loop][i] = cu.edgeWeight;
            dstArrayTabel[loop][i] = cu.destVertex.vertexID;
            dstValueTabel[loop][i] = cu.destValue;
            // if(dstValue[i] < 100)
            //     std::cout<<"["<<dstArray[i]<<","<<srcValue[i]+ewValue[i]<<","<< dstValue[i]<<"]";
            // else
            //     std::cout<<"["<<dstArray[i]<<","<<srcValue[i]+ewValue[i]<<","<< -1<<"]";
            activeNodeTabel[loop][i] = 0;
        }
        // std::cout<<std::endl;
        // std::cout<<"==================================================="<<std::endl;
        for(int i = bound[loop] ; i < arrayCount ; i++)
        {
            srcValueTabel[loop][i] = 0;
            ewValueTabel[loop][i] = 0;
            dstArrayTabel[loop][i] = 0;
            dstValueTabel[loop][i] = 0;
            activeNodeTabel[loop][i] = 0;
        }

        int CUcount = srcValueTabel[loop].size();
        int index = -1;
        memTable[loop][++index] = clCreateBuffer(context.get(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arrayCount * sizeof(int), srcValueTabel[loop].data(), nullptr);
        memTable[loop][++index] = clCreateBuffer(context.get(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arrayCount * sizeof(int), ewValueTabel[loop].data(), nullptr);
        memTable[loop][++index] = clCreateBuffer(context.get(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arrayCount * sizeof(int), dstArrayTabel[loop].data(), nullptr);
        memTable[loop][++index] = clCreateBuffer(context.get(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arrayCount * sizeof(int), dstValueTabel[loop].data(), nullptr);
        memTable[loop][++index] = clCreateBuffer(context.get(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, arrayCount * sizeof(int), activeNodeTabel[loop].data(), nullptr);
        index = -1;
        clSetKernelArg(krnls[loop].get(), 0, sizeof(cl_mem), &memTable[loop][++index]);
        clSetKernelArg(krnls[loop].get(), 1, sizeof(cl_mem), &memTable[loop][++index]);
        clSetKernelArg(krnls[loop].get(), 2, sizeof(cl_mem), &memTable[loop][++index]);
        clSetKernelArg(krnls[loop].get(), 3, sizeof(cl_mem), &memTable[loop][++index]);
        clSetKernelArg(krnls[loop].get(), 4, sizeof(cl_mem), &memTable[loop][++index]);
        clSetKernelArg(krnls[loop].get(), 5, sizeof(int), &arrayCount);
        index = -1;
        // std::cout << "stp3"<< std::endl;
        clEnqueueMigrateMemObjects(queues[loop].get(), 1, &memTable[loop][++index], 0, 0, NULL, NULL);
        clEnqueueMigrateMemObjects(queues[loop].get(), 1, &memTable[loop][++index], 0, 0, NULL, NULL);
        clEnqueueMigrateMemObjects(queues[loop].get(), 1, &memTable[loop][++index], 0, 0, NULL, NULL);
        clEnqueueMigrateMemObjects(queues[loop].get(), 1, &memTable[loop][++index], 0, 0, NULL, NULL);
        clEnqueueMigrateMemObjects(queues[loop].get(), 1, &memTable[loop][++index], 0, 0, NULL, NULL);
        // std::cout << "stp4"<< std::endl;
        //std::cout << "loop :" << loop << std::endl;
        OCL_CHECK(err, err = queues[loop].enqueueTask(krnls[loop], NULL, &eventList[loop]));
        // std::cout << "stp5"<< std::endl;
        
    }

    CUindex = 0;
    for(int loop = 0 ; loop < LoopNum ; loop++)
    {
        OCL_CHECK(err, err = eventList[loop].wait());
        cl_event endEvt;
        clEnqueueReadBuffer(queues[loop].get(), memTable[loop][3], CL_TRUE, 0, arrayCount * sizeof(int), dstValueTabel[loop].data(), 0, NULL, NULL);        
        clEnqueueReadBuffer(queues[loop].get(), memTable[loop][4], CL_TRUE, 0, arrayCount * sizeof(int), activeNodeTabel[loop].data(), 0, NULL, &endEvt);
        clWaitForEvents(1, &endEvt);
        for(int i = 0 ; i < bound[loop] ; i++)
        {
            if (!computeUnits[CUindex].destVertex.isMaster)
            {
                CUindex++;
                continue;
            }
            computeUnits[CUindex].destValue = dstValueTabel[loop][i];
            computeUnits[CUindex].destVertex.isActive = activeNodeTabel[loop][i];
            computeUnits[CUindex].srcVertex.isActive = false;
            CUindex++;
        }
        //std::cout<<std::endl;

    }
    for(int i = 0 ; i < LoopNum ; i++)
        queues[i].finish();
    //std::cout << "active node :" << avCount << std::endl;
    return 0;
}



template<typename VertexValueType, typename MessageValueType>
int
BellmanFordFPGA<VertexValueType, MessageValueType>::MSGApply_array(
    int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,MessageValueType *mValues)
{

    return 0;
}