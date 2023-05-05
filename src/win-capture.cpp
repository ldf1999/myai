
#include "sf-info.h"
#include "win-capture.h"
#include "config-info.h"

static struct win_capture win_data;

// ------------------------------------  ��ȡ��� ------------------------------------ //
static inline bool Get_Foreground_Window() {
	//��ȡ��ǰ���㴰�ھ��
	win_data.window_hwnd = GetForegroundWindow();
	if (win_data.window_hwnd == NULL) {
		std::cout << "Get_Foreground_Window: ��ȡ���㴰��ʧ��,������:" << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool Get_Specified_Window() {
	//��ȡָ�����ھ��
	win_data.window_hwnd = FindWindowA(NULL, cfg_info.Windows.win32_name.c_str());
	if (win_data.window_hwnd == NULL) {
		std::cout << "Get_Specified_Window: ��ȡָ������ʧ��,������: " << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool Win_Size_Is_Nice() {

	//��ͼԭ��С�ڴ���ԭ���,Ԥ����С�Ĵ���
	if ((global_data.origin_x) < 0) {
		std::cout << "Win_Screenshot_size: ��ǰ���㴰�ڲ���������,�ö����������Ĵ���" << std::endl;
		return false;
	}

	if ((global_data.origin_y) < 0) {
		std::cout << "Win_Screenshot_size: ��ǰ���㴰�ڲ���������,�ö����������Ĵ���" << std::endl;
		return false;
	}
	return true;
}

static inline bool Win_Screenshot_size() {

	RECT rect;
	//��ȡ������Ϣ
	if (!GetWindowRect(win_data.window_hwnd, &rect)) {
		std::cout << "Win_Screenshot_size: ��ȡ���ڿ��ʧ��" << std::endl;
		return false;
	}

	//���ڿ��
	global_data.window_width = rect.right - rect.left;
	global_data.window_height = rect.bottom - rect.top;

	//�������ĵ�
	global_data.cx = global_data.window_width * 0.5f;
	global_data.cy = global_data.window_height * 0.5f;

	//�����ͼԭ��
	global_data.origin_x = global_data.cx - (global_data.Input_Dim[2] * 0.5f);
	global_data.origin_y = global_data.cy - (global_data.Input_Dim[3] * 0.5f);

	//�����Ƿ���Ͻ�ͼ����
	if (!Win_Size_Is_Nice()) {
		return false;
	}

	//��ͼ��Ҫ���ֽ���
	win_data.size = global_data.Input_Dim[2] * global_data.Input_Dim[3] * 4;

	return true;
}

static inline bool Set_Windows_Hwnd_and_size() {

	if (cfg_info.Windows.win32_method == 1) {
		//��ȡ���㴰��
		if (!Get_Foreground_Window()) {
			return false;
		}
	}
	if (cfg_info.Windows.win32_method == 0) {
		//��ȡָ������
		if (!Get_Specified_Window()) {
			return false;
		}
	}

	//���ý�ͼ��Χ
	if (!Win_Screenshot_size()) {
		return false;
	}

	return true;
}

static inline bool Hwnd_Not_Is_Null_And_IsWindow(HWND info) {
	//�����Ϊ��
	if (info == NULL) {
		return  false;
	}

	//��һ�����ھ��
	if (!IsWindow(info)) {	//�����ڷ���0

		return  false;
	}

	return true;
}

static inline bool Init_hwnd() {
	//��ȡ���ھ���������ý�ͼ��Χ
	while (!Set_Windows_Hwnd_and_size()) {
		//�ȴ�2s
		Sleep(2000);
	}

	//�����
	if (!Hwnd_Not_Is_Null_And_IsWindow(win_data.window_hwnd)) {
		return false;
	}

	return true;
}

// ------------------------------------  ��ȡDC  ------------------------------------ //
static inline bool Get_Window_DC() {
	//��ȡ����DC(�ڴ�)
	win_data.Window_DC = GetWindowDC(win_data.window_hwnd);
	if (win_data.Window_DC == NULL) {
		return false;
	}
	return  true;
}

static inline bool Create_Compatible_DC() {
	//��������DC
	win_data.compatible_DC = CreateCompatibleDC(win_data.Window_DC);
	if (win_data.compatible_DC == NULL) {
		return false;
	}
	return true;
}

// ------------------------------------  ��ȡ����λͼ  ------------------------------------ //
static inline bool Create_Compatible_Bitmap() {
	//��������λͼ����
	win_data.BitMap = CreateCompatibleBitmap(win_data.Window_DC, global_data.Input_Dim[2], global_data.Input_Dim[3]);
	if (win_data.BitMap == NULL) {
		return false;
	}
	//��λͼ������DC(�ڴ�)
	SelectObject(win_data.compatible_DC, win_data.BitMap);
	return true;
}

static inline bool create_img() {
	//global_data.img.create(cv::Size(global_data.Input_Dim[2], global_data.Input_Dim[3]), CV_8UC4);
	//�����̶�ά�ȵĿվ���
	win_data.win_img = cv::Mat(global_data.Input_Dim[2], global_data.Input_Dim[3], CV_8UC4);
	return true;
}

static inline bool Create_Map() {
	// ------------ ˳������ ------------ //
	//��������DC
	if (!Create_Compatible_DC()) {
		std::cout << "Create_Map: ��������DCʧ��" << std::endl;
		return false;
	}
	//��������λͼ
	if (!Create_Compatible_Bitmap()) {
		std::cout << "Create_Map: ��������Mapʧ��" << std::endl;
		return false;
	}
	//�����̶�ά�ȵĿվ���
	if (!create_img()) {
		std::cout << "Create_Map: ����Matʧ��" << std::endl;
		return false;
	}
	return true;
}

// ------------------------------------  Win32_capture  ------------------------------------ //
static inline bool Get_Bitmap_Bits() {
	//��ȡλͼ����
	LONG hr = GetBitmapBits(win_data.BitMap, win_data.size, win_data.win_img.data);	//img = global_data.img
	if (hr == E_NOINTERFACE) {
		std::cout << "BitBlt_capture: �޷���ʾ��ץȡ��ɸѡ����ӵ�ͼ��" << std::endl;
		return false;
	}
	if (hr == E_POINTER) {
		std::cout << "BitBlt_capture: ָ�����" << std::endl;
		return false;
	}
	if (hr == E_UNEXPECTED) {
		std::cout << "BitBlt_capture: �������" << std::endl;
		return false;
	}
	return true;
}

static inline bool Window_again_should_init() {
	//���ھ���Ƿ�Ϊ����ָ��һ������
	if (!Hwnd_Not_Is_Null_And_IsWindow(win_data.window_hwnd)) {

		std::cout << "BitBlt_capture: ���ڶ�ʧ�����³�ʼ������" << std::endl;
		//���³�ʼ�����
		if (!Init_hwnd()) {
			std::cout << "Window_again_should_init: ���³�ʼ�����ʧ��" << std::endl;
			return false;
		}
		//���»�ȡDC
		if (!Get_Window_DC()) {
			std::cout << "Window_again_should_init: ���»�ȡ����DCʧ��" << std::endl;
		}
		//���´�������DC
		if (!Create_Compatible_DC()) {
			std::cout << "Window_again_should_init: ���´�������DCʧ��" << std::endl;
			return false;
		}
		//���´�������λͼ
		if (!Create_Compatible_Bitmap()) {
			std::cout << "Window_again_should_init: ���´�������Mapʧ��" << std::endl;
			return false;
		}
	}
	return true;
}

static inline bool Check_Is_Foreground_Window() {
	// ------------ δ��ɣ�δʹ�� ------------ //
	//��ȡ��ǰ�ö�����
	HWND temp_hwnd = GetForegroundWindow();
	//��ǰ�ö����ھ�����ͼ���ھ����һ��
	if (temp_hwnd != win_data.window_hwnd){
		return true;
	}
	return true;
}

static inline bool BitBlt_() {
	//��ת
	LONG hr = BitBlt(win_data.compatible_DC, 0, 0, global_data.Input_Dim[2], global_data.Input_Dim[3], win_data.Window_DC, global_data.origin_x, global_data.origin_y, SRCCOPY);
	if (FAILED(hr)) {
		std::cout << "BitBlt_capture: BitBltʧ��,������: "<<GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool BitBlt_capture() {

	//��鴰���Ƿ�ʧand���³�ʼ��
	if (!Window_again_should_init()) {
		std::cout << "BitBlt_capture: ���³�ʼ������ʧ��" << std::endl;
		return false;
	}

	//����Ƿ��ö�����
	//if (!Check_Is_Foreground_Window()){
	//}

	//��ת����ͼƬ������DC
	if (!BitBlt_()){
		std::cout << "BitBlt_capture: BitBltʧ��" << std::endl;
		return false;
	}
	//��ȡλͼ����
	if (!Get_Bitmap_Bits()) {
		std::cout << "BitBlt_capture: Get_Bitmap_Bitsʧ��" << std::endl;
		return false;
	}
	//4-> ˳����Get_Bitmap_Bits���λͼ����ȫ��ͼƬ��
	cv::cvtColor(win_data.win_img, global_data.img, cv::COLOR_BGRA2BGR);

	return true;
}

// ------------------------------------  Win32_Free  ------------------------------------ //
bool Free_BitBlt() {

	if (win_data.Window_DC) {
		DeleteDC(win_data.Window_DC);
	}
	if (win_data.compatible_DC) {
		DeleteDC(win_data.compatible_DC);
	}
	if (win_data.BitMap) {
		DeleteObject(win_data.BitMap);
	}
	if (win_data.window_hwnd) {
		CloseHandle(win_data.window_hwnd);
	}

	return true;
}

// ------------------------------------  Win32_Init  ------------------------------------ //
bool Init_BitBlt() {

	//��ȡ���ھ���������ý�ȡ��Χ
	if (!Init_hwnd()) {
		std::cout << "Init_BitBlt: ��ʼ�����ʧ��" << std::endl;
		return false;
	}
	//��ȡ����DC
	if (!Get_Window_DC()) {
		std::cout << "Init_BitBlt: ��ȡ����DCʧ��" << std::endl;
	}
	//����Map��Դ
	if (!Create_Map()) {
		std::cout << "Init_BitBlt: ��ʼ��λͼʧ��" << std::endl;
		return false;
	}

	global_data.capture_map = BitBlt_capture;
	global_data.capture_free = Free_BitBlt;

	return true;
}
