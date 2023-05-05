#pragma once
#include<opencv2/opencv.hpp>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_5.h>


bool Init_dxgi();
bool Free_dxgi();

struct dxgi_info{
	cv::Rect rect;
	UINT output = 0;	//��ʾ������
	ID3D11Device* device = nullptr;	//ID3D11 �豸
	ID3D11DeviceContext* Context = nullptr;	//ID3D11 �豸������
	IDXGIDevice2* DxgiDevice2 = nullptr;
	IDXGIAdapter* DxgiAdapter = nullptr;
	IDXGIOutput* DxgiOutput = nullptr;
	IDXGIOutput1* DxgiOutput1 = nullptr;

	IDXGIOutputDuplication* desktopDupl = nullptr;
	DXGI_OUTDUPL_FRAME_INFO frameInfo;	//DXGI ���֡��Ϣ
	IDXGIResource* desktopResource = nullptr;	//��������
	ID3D11Texture2D* acquiredDesktopImage = nullptr;	//��ȡ����������
	D3D11_TEXTURE2D_DESC dataDesc = {NULL};
	ID3D11Texture2D* copyDesktop = nullptr;	//������������
	D3D11_MAPPED_SUBRESOURCE mappedResource;	//D3D11 GPUӳ�䶼cpu
};