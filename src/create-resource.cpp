#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>


#include "sf-info.h"
#include"config-info.h"
#include "create-resource.h"
#include "process-info.h"

static Logger gLogger;

//全局结构体
struct trt_interface trt;			//trt接口
struct predict_data pred_data;		//推理所需要的参数

static bool load_engien(char*& trtModelStream, size_t& size) {
	// 反序列化 .engine 并运行推理
	std::ifstream file(cfg_info.Pred.engine_path.c_str(), std::ios::binary);      //engine路径  ofstream是从内存到硬盘，ifstream是从硬盘到内存
	if (!file.good()) {   //判断当前流的状态（读写正常（即符合读取和写入的类型)，没有文件末尾）
		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cerr << "读取 " << cfg_info.Pred.engine_path << " 错误, 请检查文件和路径." << std::endl;
		std::cout << "-------------------------------------------------------------" << std::endl;
		return false;
	}
	file.seekg(0, file.end);            //seekg()是对输入流的操作,从文件末尾开始计算偏移量
	size = file.tellg();                //得到file的大小
	file.seekg(0, file.beg);            //从文件头开始计算偏移量
	trtModelStream = new char[size];    //创建一个堆区由trtModelStream指针维护
	assert(trtModelStream);             //创建成功？
	file.read(trtModelStream, size);    //读取流
	file.close();                       //关闭

	return true;
}

static bool init_trt(char*& trtModelStream, size_t& size) {
	//创建runtime接口
	trt.runtime = nvinfer1::createInferRuntime(gLogger);
	if (trt.runtime == NULL) {
		std::cout << "init_trt:创建runtime失败" << std::endl;
		return false;
	}
	//创建engine推理引擎
	trt.engine = trt.runtime->deserializeCudaEngine(trtModelStream, size);
	if (trt.engine == NULL) {
		std::cout << "init_trt:创建engine失败" << std::endl;
		return false;
	}
	//获取上下文
	trt.context = trt.engine->createExecutionContext();
	if (trt.context == NULL) {
		std::cout << "init_trt:创建context失败" << std::endl;
		return false;
	}

	return true;
}

//计算Dim大小
static inline int volume(nvinfer1::Dims dims) {
	int temp = 1;
	for (int i = 0; i < dims.nbDims; i++) 
		temp *= dims.d[i];
	
	return temp;
}

//初始化推理和后处理需要的数据
static  bool init_trt_data() {
	//类别数量
	Process_data.Num_Classes = trt.engine->getBindingDimensions(trt.engine->getNbBindings() - 1).d[2] - 5;
	if (!Process_data.Num_Classes) {
		std::cout << "init_trt_data:获取类别数量失败" << std::endl;
		return false;
	}

	//输入维度
	pred_data.ip_Dim = trt.engine->getBindingDimensions(0);
	if (!pred_data.ip_Dim.nbDims) {
		std::cout << "init_trt_data:获取输入维度失败" << std::endl;
		return false;
	}

	//输出维度 yolov5是4个输出，只需要最后一个输出的维度
	pred_data.op_Dim = trt.engine->getBindingDimensions(trt.engine->getNbBindings() - 1);
	if (!pred_data.op_Dim.nbDims) {
		std::cout << "init_trt_data:获取输入维度失败" << std::endl;
		return false;
	}

	//申请的显存数量
	pred_data.buffer_size.resize(trt.engine->getNbBindings());

	//申请的内存大小
	pred_data.intput_size = volume(pred_data.ip_Dim);
	pred_data.output_size = volume(pred_data.op_Dim);

	//输入/输出维度放到全局结构体，以供其他程序使用
	for (int i = 0; i < pred_data.ip_Dim.nbDims; i++) {
		global_data.Input_Dim[i] = pred_data.ip_Dim.d[i];
		//std::cout << global_data.Input_Dim[i] << std::endl;
	}
	for (int i = 0; i < pred_data.op_Dim.nbDims; i++) {
		global_data.Output_Dim[i] = pred_data.op_Dim.d[i];
		//std::cout << global_data.Output_Dim[i] << std::endl;
	}
	return true;
}

static inline bool create_resource() {
	//创建显存
	for (size_t i = 0; i < trt.engine->getNbBindings(); i++) {
		//维度
		nvinfer1::Dims dims = trt.engine->getBindingDimensions(i);
		//数据类型
		nvinfer1::DataType dtype = trt.engine->getBindingDataType(i);
		//大小
		int64_t total_size = static_cast<unsigned long long>(1) * volume(dims) * sizeof(dtype);
		//申请显存的大小
		pred_data.buffer_size[i] = total_size;
		//申请显存
		CUDA_CHECK(cudaMalloc(&pred_data.buffers_ptr[i], total_size))
	}

	//创建cuda流
	CUDA_CHECK(cudaStreamCreate(&trt.stream));

	//创建输入内存
	pred_data.intput = new float[pred_data.intput_size];
	if (!pred_data.intput) {
		std::cout << "create_resource: 创建输入内存失败" << std::endl;
		return false;
	}
	//创建输出内存
	pred_data.output = new float[pred_data.output_size];
	if (!pred_data.output) {
		std::cout << "create_resource: 创建输出内存失败" << std::endl;
		return false;
	}

	return true;
}


static bool create_runtime(char*& trtModelStream, size_t& size) {
	//初始化接口
	if (!init_trt(trtModelStream, size)) {
		std::cout << "create_runtime:创建trt对象失败" << std::endl;
		return false;
	}
	//加载完engine模型，立即缩放创建的堆区
	delete[] trtModelStream;

	//初始化需要的数据
	if (!init_trt_data()) {
		std::cout << "create_runtime:初始化数据大小数据失败" << std::endl;
		return false;
	}
	//创建资源
	if (!create_resource()) {
		std::cout << "create_runtime:创建内存/显存失败资源失败" << std::endl;
		return false;
	}
	return true;
}

bool Init_CUDA() {

	char* trtModelStream = nullptr;     //创建流
	size_t size = 0;
	//加载engine
	if (!load_engien(trtModelStream, size)) {
		return false;
	}
	//创建资源
	if (!create_runtime(trtModelStream, size)) {
		return false;
	}
	std::cout << "Init CUDA resource PASS..." << std::endl;
	return true;
}

bool Free_CUDA() {
	if (trt.stream) {
		cudaStreamDestroy(trt.stream);
	}

	for (int i = 0; i < trt.engine->getNbBindings(); i++) {   //循环释放
		if (pred_data.buffers_ptr[i]) {
			CUDA_CHECK(cudaFree(pred_data.buffers_ptr[i]));
		}
	}
	if (trt.context) {
		trt.context->destroy();
	}
	if (trt.engine) {
		trt.engine->destroy();
	}
	if (trt.runtime) {
		trt.runtime->destroy();
	}
	if (pred_data.intput != NULL) {
		delete[] pred_data.intput;
	}
	if (pred_data.output != NULL) {
		delete[] pred_data.output;
	}
	return true;
}