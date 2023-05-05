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

    int intput_size;            //�����ڴ��С
    int output_size;            //����ڴ��С
    nvinfer1::Dims  ip_Dim;     //����ά��
    nvinfer1::Dims  op_Dim;     //���ά��
    std::vector<int64_t> buffer_size;       //�����ڴ��С 
    void* buffers_ptr[5];       //GPUָ��
    float* intput;              //�����ڴ�ָ��
    float* output;              //����ڴ�ָ��
};


bool Init_CUDA();
bool Free_CUDA();