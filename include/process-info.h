#pragma once


bool Init_frame();


struct GridAndStride{
	int grid0;
	int grid1;
	int stride;
};

struct Porocess_info {
	std::vector<cv::Rect> boxes;	 //��������
	std::vector<int> Classes;        //�������
	std::vector<float> confidences;  //���Ŷ�����
	std::vector<int> indices;		 //��������Ϣ
	std:: vector<float> recent;		 //���Ŀ��
	cv::Rect target_xywh;				 //���Ŀ���xywh	

	std::vector<int> strides = { 8, 16, 32 };       //����
	std::vector<GridAndStride> grid_strides;    //����
	int Num_Classes;

};	

struct alignas(float) BBOX{
	cv::Rect_<float> box;
	float confidence;  // bbox_conf * cls_conf        ����ع�����Ŷ� * �������Ŷ�
	float index; //���
};