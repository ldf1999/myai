#pragma once

#include<windows.h>

DWORD WINAPI aim_func();	//����������

struct PID {

	float SetSpeed = 0;			//��¼�����ƶ�����
	float Error = 0;			//����һ���ƶ������
	float Err_Next = 0;			//
	float ActualSpeed = 0;		//��һ�ε��ƶ���������
	float IncrementSpeed = 0;	//�������ƶ�����
	float Err_last = 0;			//�������ƶ�����
	int Sample_frame = 0;		//����֡

};