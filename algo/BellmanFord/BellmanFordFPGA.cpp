#include "BellmanFordFPGA.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

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
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    context = context_tmp;
    queue = q;
    std::string binaryFile = xcl::find_binary_file(device_name, "gs_top");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    OCL_CHECK(err, cl::Program program_tmp(context, devices, bins, NULL, &err));
    program = program_tmp;
    OCL_CHECK(err, cl::Kernel kernel(program, "gs_top", &err));
    krnl = kernel;
}

template<typename VertexValueType, typename MessageValueType>
int
BellmanFordFPGA<VertexValueType, MessageValueType>::MSGApply_array(
    int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,MessageValueType *mValues)
{
    std::vector<int,aligned_allocator<int>> srcValue(computeUnitCount);
    std::vector<int,aligned_allocator<int>> ewValue(computeUnitCount);
    std::vector<int,aligned_allocator<int>> dstArray(computeUnitCount);
    std::vector<int,aligned_allocator<int>> dstValue(computeUnitCount+1);
    std::vector<int,aligned_allocator<int>> activeNode(computeUnitCount,0);

    for(int i = 0 ; i < computeUnitCount ; i++)
    {
        auto cu = computeUnits[i];
        srcValue[i] = cu.srcValue;
        ewValue[i] = cu.edgeWeight;
        dstArray[i] = cu.destVertex.vertexID;
        dstValue[i] = cu.destValue;
    }

    cl_mem buffer_1 = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, srcValue.size() * sizeof(int), nullptr, nullptr);
    cl_mem buffer_2 = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, ewValue.size() * sizeof(int), nullptr, nullptr);
    cl_mem buffer_3 = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, dstArray.size() * sizeof(int), nullptr, nullptr);
    cl_mem buffer_4 = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, dstValue.size() * sizeof(int), nullptr, nullptr);
    cl_mem buffer_5 = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, activeNode.size() * sizeof(int), nullptr, nullptr);

    clSetKernelArg(krnl.get(), 0, sizeof(cl_mem), &buffer_1);
    clSetKernelArg(krnl.get(), 1, sizeof(cl_mem), &buffer_2);
    clSetKernelArg(krnl.get(), 2, sizeof(cl_mem), &buffer_3);
    clSetKernelArg(krnl.get(), 3, sizeof(cl_mem), &buffer_4);
    clSetKernelArg(krnl.get(), 4, sizeof(cl_mem), &buffer_5);
    clSetKernelArg(krnl.get(), 5, sizeof(int), &computeUnitCount);

    clEnqueueWriteBuffer(queue.get(), buffer_1, CL_TRUE, 0, srcValue.size() * sizeof(int), srcValue.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), buffer_2, CL_TRUE, 0, ewValue.size() * sizeof(int), ewValue.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), buffer_3, CL_TRUE, 0, dstArray.size() * sizeof(int), dstArray.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), buffer_4, CL_TRUE, 0, dstValue.size() * sizeof(int), dstValue.data(),0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue.get(), buffer_5, CL_TRUE, 0, activeNode.size() * sizeof(int), activeNode.data(), 0, nullptr, nullptr);

    clEnqueueMigrateMemObjects(queue.get(), 1, &buffer_1, 0, 0, NULL, NULL);
    clEnqueueMigrateMemObjects(queue.get(), 1, &buffer_2, 0, 0, NULL, NULL);
    clEnqueueMigrateMemObjects(queue.get(), 1, &buffer_3, 0, 0, NULL, NULL);
    clEnqueueMigrateMemObjects(queue.get(), 1, &buffer_4, 0, 0, NULL, NULL);
    clEnqueueMigrateMemObjects(queue.get(), 1, &buffer_5, 0, 0, NULL, NULL);

    cl::Event event;
    OCL_CHECK(err, err = queue.enqueueTask(krnl, NULL, &event));
    OCL_CHECK(err, err = event.wait());

    clEnqueueReadBuffer(queue.get(), buffer_4, CL_TRUE, 0, dstValue.size() * sizeof(int), dstValue.data(), 0, NULL, NULL);        
    clEnqueueReadBuffer(queue.get(), buffer_5, CL_TRUE, 0, activeNode.size() * sizeof(int), activeNode.data(), 0, NULL, NULL);
    
    for(int i = 0 ; i < computeUnitCount ; i++)
    {
        computeUnits[i].destValue = dstValue[i];
        computeUnits[i].destVertex.isActive = activeNode[i];
    }

    return dstValue[computeUnitCount];
}



template<typename VertexValueType, typename MessageValueType>
int
BellmanFordFPGA<VertexValueType, MessageValueType>::MSGGenMerge_array(
    int computeUnitCount, ComputeUnit<VertexValueType> *computeUnits,MessageValueType *mValues)
{

    return 0;
}