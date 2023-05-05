#pragma once
#include<Windows.h>
#include "logging.h"
#include<opencv2/opencv.hpp>
#include<chrono>

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }

//ȫ�ֺ���
bool Thread_Should_Stop();

struct global_info {

    cv::Mat img;                        //��ȡ��ͼƬ
    int origin_x, origin_y,cx,cy;       //ԭ�㣬���ĵ�
    int window_width, window_height;    //���ڿ��
    int Input_Dim[4],Output_Dim[3];     //ȫ�� ����/���
    bool (*capture_map)();              //capture������ַ
    bool (*capture_free)();             //capture�ͷŵ�ַ

    bool (*Pre_process)(cv::Mat& img);  //Ԥ��������ַ
    bool (*Post_poress)();              //��������ַ

    bool (*Move_R)(int, int);           //�ƶ�������ַ
    bool (*Move_free)();                //�ͷŵ�ַ
};

struct Time_info {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;

    std::chrono::system_clock::time_point capture_start;
    std::chrono::system_clock::time_point capture_end;

    std::chrono::system_clock::time_point Inference_start;
    std::chrono::system_clock::time_point Inference_end;
};

//ȫ�ֱ���
extern struct Config_Info cfg_info;         //cfg����
extern struct global_info global_data;      //ȫ�ֽṹ��
extern struct trt_interface trt;			//trt�ӿ�
extern struct predict_data pred_data;		//��������Ҫ�Ĳ���
extern struct Porocess_info Process_data;   //��ܴ��������

extern HANDLE Aim_Event;                         //�¼�
extern HANDLE Dynamic_Event;                         //�¼�


