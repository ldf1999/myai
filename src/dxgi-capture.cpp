
#include "sf-info.h"
#include"dxgi-capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")


struct dxgi_info dxgi_data;

static bool Desc_Should_Init = false;       //Desc�Ƿ���Ҫ��ʼ��


static inline bool Init_Device() {
	D3D_FEATURE_LEVEL FeatureLevel;

	//����d3d11�豸
	HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
		nullptr, 0, NULL, NULL, D3D11_SDK_VERSION,
		&dxgi_data.device, &FeatureLevel, &dxgi_data.Context);
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool GetDevice2() {
	//�����豸�����豸2
	HRESULT hr = dxgi_data.device->QueryInterface(__uuidof(IDXGIDevice2), (void**)&dxgi_data.DxgiDevice2);
	if (FAILED(hr)) {
		return false;
	}
	//�������֡���ӳ�
	dxgi_data.DxgiDevice2->SetMaximumFrameLatency(1);
	return true;
}

static inline bool Get_DXGI_Adapter() {
	//��ȡ�豸������
	HRESULT hr = dxgi_data.DxgiDevice2->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&dxgi_data.DxgiAdapter));
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_DxgiOutput() {
	//ö����ʾ��
	HRESULT hr = dxgi_data.DxgiAdapter->EnumOutputs(dxgi_data.output, &dxgi_data.DxgiOutput);
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_Output1_Interface() {
	//��ȡ��ʾ���ӿ�
	HRESULT hr = dxgi_data.DxgiOutput->QueryInterface(__uuidof(dxgi_data.DxgiOutput1), reinterpret_cast<void**>(&dxgi_data.DxgiOutput1));	//��ѯ�ӿ�
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_desk_Duplicate() {
	//��ȡ���渱����Desk��Ϣ
	HRESULT hr = dxgi_data.DxgiOutput1->DuplicateOutput(dxgi_data.device, &dxgi_data.desktopDupl);
	//����win10���ִ˴���δ֪ԭ��
	if (hr == DXGI_ERROR_UNSUPPORTED){
		std::cout << "Get_desk_Duplicate: �豸����������֧������Ĺ���" << std::endl;
		return false;
	}
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Init_Interface() {
	if (!GetDevice2()) {
		std::cout << "Init_Interface: ��ȡDevice2ʧ��" << std::endl;
		return false;
	}
	if (!Get_DXGI_Adapter()) {
		std::cout << "Init_Interface: ��ȡ�Կ�������ʧ��" << std::endl;
		return false;
	}
	if (!Get_DxgiOutput()) {
		std::cout << "Init_Interface: ��ȡ��ʾ������ʧ��" << std::endl;
		return false;
	}
	if (!Get_Output1_Interface()) {
		std::cout << "Init_Interface: ��ȡOutput1�ӿ�ʧ��" << std::endl;
		return false;
	}
	if (!Get_desk_Duplicate()) {
		std::cout << "Init_Interface: ��ȡ���渱���ӿ�ʧ��" << std::endl;
		return false;
	}
	return true;
}

static inline bool Get_Next_Frame() {

	//��ȡ�󻺳�
	HRESULT hr = dxgi_data.desktopDupl->AcquireNextFrame(0, &dxgi_data.frameInfo, &dxgi_data.desktopResource);
	//�����ޱ仯ʱ��dxgi���ߣ��������κβ�����true����
	if (hr == DXGI_ERROR_WAIT_TIMEOUT) {		
		//std::cout << "�����ޱ仯" << std::endl;
		return true;
	}

	//���AcquireNextFrame ������ʧ,��Ҫ���³�ʼ�� desktopDupl��������cf������Ϸ
	if (FAILED(hr)) {	
		std::cout << "Get_Next_Frame: desktopDupl��ʧ�����³�ʼ��" << std::endl;
		if (!Get_desk_Duplicate()) {
			std::cout << "Get_Next_Frame: desktopDupl���³�ʼ��ʧ��" << std::endl;
			return false;
		}else {
			std::cout << "Get_Next_Frame: desktopDupl�س���ʼ���ɹ�" << std::endl;
			Desc_Should_Init = true;	//����Desk��Ϣ
			return true;
		}
	}
	return true;
}

static inline bool Get_Frame_prt() {
	//��ȡ�󻺳�ָ�� 
	HRESULT hr = dxgi_data.desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&dxgi_data.acquiredDesktopImage));
	if (FAILED(hr)) {
		return false;
	}

	return true;
}

static bool Init_Desc() {
	//��ʼ�����渱����Ϣ
	dxgi_data.acquiredDesktopImage->GetDesc(&dxgi_data.dataDesc);
	dxgi_data.dataDesc.Usage = D3D11_USAGE_STAGING;
	dxgi_data.dataDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	dxgi_data.dataDesc.BindFlags = 0;
	dxgi_data.dataDesc.MiscFlags = 0;
	dxgi_data.dataDesc.MipLevels = 1;
	dxgi_data.dataDesc.SampleDesc.Count = 1;
	return true;
}

static inline bool Create_Tex() {
	//�������ݵ�2d����
	dxgi_data.device->CreateTexture2D(&dxgi_data.dataDesc, NULL, &dxgi_data.copyDesktop);
	if (dxgi_data.copyDesktop == NULL) {
		return false;
	}
	return true;
}

static inline bool Unmap_and_Release() {
	dxgi_data.Context->Unmap(dxgi_data.copyDesktop, NULL);	//ȡ��ӳ��
	dxgi_data.copyDesktop->Release();		//�ͷ�����
	dxgi_data.desktopDupl->ReleaseFrame();	//�ͷź󻺳�
	return true;
}

static inline bool DXG_Capture_Map() {

	//��ȡ�󻺳�
	if (!Get_Next_Frame()) {	//0ms
		return false;
	}
	//��ȡ�󻺳��ָ��
	if (!Get_Frame_prt()) {	//0ms
		return false;
	}
	//ֻ��ʼ��1��
	if (Desc_Should_Init) {	//0ms
		if (Init_Desc()) {
			Desc_Should_Init = false;
		}
	}
	//����2d����
	if (!Create_Tex()) {	//0ms
		return false;
	}

	//copy 0ms
	dxgi_data.Context->CopyResource(dxgi_data.copyDesktop, dxgi_data.acquiredDesktopImage);

	//������ӳ�䵽cpu 1ms
	dxgi_data.Context->Map(dxgi_data.copyDesktop, 0, D3D11_MAP_READ, 0, &dxgi_data.mappedResource);

	//תΪMat 0ms
	global_data.img = cv::Mat(int(dxgi_data.dataDesc.Height), int(dxgi_data.dataDesc.Width), CV_8UC4, dxgi_data.mappedResource.pData)(dxgi_data.rect);

	//4->3	1-3ms
	cv::cvtColor(global_data.img, global_data.img, cv::COLOR_BGRA2BGR);

	if (!Unmap_and_Release()) {	//0ms
		std::cout << "Unmap_and_Release: �ͷŴ���" << std::endl;
		return false;
	}

	return true;
}

static inline bool DXGI_Screenshot_size() {

	//��ȡ�󻺳�
	if (Get_Next_Frame()) {	//0ms
		dxgi_data.desktopDupl->ReleaseFrame();	//�ͷź󻺳�
	}
	//�󻺳��ָ��
	if (Get_Frame_prt()) {	//0ms
		dxgi_data.acquiredDesktopImage->Release();
	}

	//��ʼ��Desk
	if (!Init_Desc()) {
		std::cout << "Create_Desc����" << std::endl;
	}

	//��Ļ��С
	global_data.window_width = int(dxgi_data.dataDesc.Width);
	global_data.window_height = int(dxgi_data.dataDesc.Height);

	//��Ļ���ĵ�
	global_data.cx = int(dxgi_data.dataDesc.Width * 0.5f);
	global_data.cy = int(dxgi_data.dataDesc.Height * 0.5f);

	//��ͼԭ��
	global_data.origin_x = global_data.cx - int(global_data.Input_Dim[2] * 0.5f);
	global_data.origin_y = global_data.cy - int(global_data.Input_Dim[3] * 0.5f);

	//������ͼ��Χ
	dxgi_data.rect = cv::Rect(global_data.origin_x, global_data.origin_y, global_data.Input_Dim[2], global_data.Input_Dim[3]);

	return true;
}

bool Free_dxgi() {

	if (dxgi_data.device) {
		dxgi_data.device->Release();	//�ͷ�
		dxgi_data.device = nullptr;	//��Ұ
	}
	if (dxgi_data.Context) {
		dxgi_data.Context->Release();
		dxgi_data.Context = nullptr;
	}

	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//�ͷ�
		dxgi_data.DxgiDevice2 = nullptr;	//��Ұ
	}
	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//�ͷ�
		dxgi_data.DxgiDevice2 = nullptr;	//��Ұ
	}
	if (dxgi_data.DxgiAdapter) {
		dxgi_data.DxgiAdapter->Release();
		dxgi_data.DxgiAdapter = nullptr;
	}
	if (dxgi_data.DxgiOutput) {
		dxgi_data.DxgiOutput->Release();
		dxgi_data.DxgiOutput = nullptr;
	}
	if (dxgi_data.DxgiOutput1) {
		dxgi_data.DxgiOutput1->Release();
		dxgi_data.DxgiOutput1 = nullptr;
	}
	if (dxgi_data.desktopDupl) {
		dxgi_data.desktopDupl->Release();
		dxgi_data.desktopDupl = nullptr;
	}
	if (dxgi_data.desktopResource) {
		dxgi_data.desktopResource->Release();
		dxgi_data.desktopResource = nullptr;
	}

	if (dxgi_data.acquiredDesktopImage) {
		dxgi_data.acquiredDesktopImage->Release();
		dxgi_data.acquiredDesktopImage = nullptr;
	}

	//Խ���������
	//if (dxgi_data.copyDesktop) {
	//	dxgi_data.copyDesktop->Release();
	//	dxgi_data.copyDesktop = nullptr;
	//}

	return true;
}

bool Init_dxgi() {
	//����d3d11�豸
	if (!Init_Device()) {
		std::cout << "Init_dxgi: ����D3D11�豸ʧ��" << std::endl;
		return false;
	}

	//����d3d�ӿ�
	if (!Init_Interface()) {
		std::cout << "Init_dxgi: ��ʼ���ӿ�ʧ��" << std::endl;
		return false;
	}

	//���ý�ͼ��С
	if (!DXGI_Screenshot_size()) {
		std::cout << "Init_dxgi: ���ý�ͼ��Χʧ��" << std::endl;
		return false;
	}

	//��־��Ҫ��ʼ��Desc
	Desc_Should_Init = true;

	//������ַ
	//global_data.capture_map�洢����DXG_Capture_Map�������ڴ��ַ
	//global_data.capture_map() ��Ч�� DXG_Capture_Map()
	global_data.capture_map = DXG_Capture_Map;
	global_data.capture_free = Free_dxgi;

	return true;
}

