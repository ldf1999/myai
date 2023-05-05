#pragma once
#include<vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>


struct trt_interface {
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;

};

struct predict_data{

    int intput_size;            //输入内存大小
    int output_size;            //输出内存大小
    nvinfer1::Dims  ip_Dim;     //输入维度
    nvinfer1::Dims  op_Dim;     //输出维度
    std::vector<int64_t> buffer_size;       //申请内存大小 
    void* buffers_ptr[5];       //GPU指针
    float* intput;              //输入内存指针
    float* output;              //输出内存指针
};


bool Init_CUDA();
bool Free_CUDA();