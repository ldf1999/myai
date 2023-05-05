#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>


#include "sf-info.h"
#include"config-info.h"
#include "create-resource.h"
#include "process-info.h"

static Logger gLogger;

//ȫ�ֽṹ��
struct trt_interface trt;			//trt�ӿ�
struct predict_data pred_data;		//��������Ҫ�Ĳ���

static bool load_engien(char*& trtModelStream, size_t& size) {
	// �����л� .engine ����������
	std::ifstream file(cfg_info.Pred.engine_path.c_str(), std::ios::binary);      //engine·��  ofstream�Ǵ��ڴ浽Ӳ�̣�ifstream�Ǵ�Ӳ�̵��ڴ�
	if (!file.good()) {   //�жϵ�ǰ����״̬����д�����������϶�ȡ��д�������)��û���ļ�ĩβ��
		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cerr << "��ȡ " << cfg_info.Pred.engine_path << " ����, �����ļ���·��." << std::endl;
		std::cout << "-------------------------------------------------------------" << std::endl;
		return false;
	}
	file.seekg(0, file.end);            //seekg()�Ƕ��������Ĳ���,���ļ�ĩβ��ʼ����ƫ����
	size = file.tellg();                //�õ�file�Ĵ�С
	file.seekg(0, file.beg);            //���ļ�ͷ��ʼ����ƫ����
	trtModelStream = new char[size];    //����һ��������trtModelStreamָ��ά��
	assert(trtModelStream);             //�����ɹ���
	file.read(trtModelStream, size);    //��ȡ��
	file.close();                       //�ر�

	return true;
}

static bool init_trt(char*& trtModelStream, size_t& size) {
	//����runtime�ӿ�
	trt.runtime = nvinfer1::createInferRuntime(gLogger);
	if (trt.runtime == NULL) {
		std::cout << "init_trt:����runtimeʧ��" << std::endl;
		return false;
	}
	//����engine��������
	trt.engine = trt.runtime->deserializeCudaEngine(trtModelStream, size);
	if (trt.engine == NULL) {
		std::cout << "init_trt:����engineʧ��" << std::endl;
		return false;
	}
	//��ȡ������
	trt.context = trt.engine->createExecutionContext();
	if (trt.context == NULL) {
		std::cout << "init_trt:����contextʧ��" << std::endl;
		return false;
	}

	return true;
}

//����Dim��С
static inline int volume(nvinfer1::Dims dims) {
	int temp = 1;
	for (int i = 0; i < dims.nbDims; i++) 
		temp *= dims.d[i];
	
	return temp;
}

//��ʼ������ͺ�����Ҫ������
static  bool init_trt_data() {
	//�������
	Process_data.Num_Classes = trt.engine->getBindingDimensions(trt.engine->getNbBindings() - 1).d[2] - 5;
	if (!Process_data.Num_Classes) {
		std::cout << "init_trt_data:��ȡ�������ʧ��" << std::endl;
		return false;
	}

	//����ά��
	pred_data.ip_Dim = trt.engine->getBindingDimensions(0);
	if (!pred_data.ip_Dim.nbDims) {
		std::cout << "init_trt_data:��ȡ����ά��ʧ��" << std::endl;
		return false;
	}

	//���ά�� yolov5��4�������ֻ��Ҫ���һ�������ά��
	pred_data.op_Dim = trt.engine->getBindingDimensions(trt.engine->getNbBindings() - 1);
	if (!pred_data.op_Dim.nbDims) {
		std::cout << "init_trt_data:��ȡ����ά��ʧ��" << std::endl;
		return false;
	}

	//������Դ�����
	pred_data.buffer_size.resize(trt.engine->getNbBindings());

	//������ڴ��С
	pred_data.intput_size = volume(pred_data.ip_Dim);
	pred_data.output_size = volume(pred_data.op_Dim);

	//����/���ά�ȷŵ�ȫ�ֽṹ�壬�Թ���������ʹ��
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
	//�����Դ�
	for (size_t i = 0; i < trt.engine->getNbBindings(); i++) {
		//ά��
		nvinfer1::Dims dims = trt.engine->getBindingDimensions(i);
		//��������
		nvinfer1::DataType dtype = trt.engine->getBindingDataType(i);
		//��С
		int64_t total_size = static_cast<unsigned long long>(1) * volume(dims) * sizeof(dtype);
		//�����Դ�Ĵ�С
		pred_data.buffer_size[i] = total_size;
		//�����Դ�
		CUDA_CHECK(cudaMalloc(&pred_data.buffers_ptr[i], total_size))
	}

	//����cuda��
	CUDA_CHECK(cudaStreamCreate(&trt.stream));

	//���������ڴ�
	pred_data.intput = new float[pred_data.intput_size];
	if (!pred_data.intput) {
		std::cout << "create_resource: ���������ڴ�ʧ��" << std::endl;
		return false;
	}
	//��������ڴ�
	pred_data.output = new float[pred_data.output_size];
	if (!pred_data.output) {
		std::cout << "create_resource: ��������ڴ�ʧ��" << std::endl;
		return false;
	}

	return true;
}


static bool create_runtime(char*& trtModelStream, size_t& size) {
	//��ʼ���ӿ�
	if (!init_trt(trtModelStream, size)) {
		std::cout << "create_runtime:����trt����ʧ��" << std::endl;
		return false;
	}
	//������engineģ�ͣ��������Ŵ����Ķ���
	delete[] trtModelStream;

	//��ʼ����Ҫ������
	if (!init_trt_data()) {
		std::cout << "create_runtime:��ʼ�����ݴ�С����ʧ��" << std::endl;
		return false;
	}
	//������Դ
	if (!create_resource()) {
		std::cout << "create_runtime:�����ڴ�/�Դ�ʧ����Դʧ��" << std::endl;
		return false;
	}
	return true;
}

bool Init_CUDA() {

	char* trtModelStream = nullptr;     //������
	size_t size = 0;
	//����engine
	if (!load_engien(trtModelStream, size)) {
		return false;
	}
	//������Դ
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

	for (int i = 0; i < trt.engine->getNbBindings(); i++) {   //ѭ���ͷ�
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