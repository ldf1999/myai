#include <math.h>
#include <windows.h>

#include "msdk.h"
#include"sf-info.h"			//ʹ��ȫ��config_info
#include "config-info.h"	//�����ṹ��
#include"move-info.h"		

static struct Logitech_info Log_info;		//�޼��ӿ�
static struct KmBox_info Km_info;			//kmbox�ӿ�
static struct FeiOrYi_info FOY_info;		//������/�׼���ӿ�
static struct Send_info Sen_info;			//Sendinput�ӿ�


// -----------------------------------   �޼��ƶ�  -----------------------------------
static inline bool Get_NT_status(PCWSTR device_name) {


	UNICODE_STRING name;
	OBJECT_ATTRIBUTES attr{};
	RtlInitUnicodeString(&name, device_name);
	InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

	//��.Net
	Log_info.LG_status = NtCreateFile(&Log_info.LG_input, GENERIC_WRITE | SYNCHRONIZE, &attr, &Log_info.LG_io, 0,
		FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);
	if (!(NT_SUCCESS(Log_info.LG_status))) {
		return false;
	}

	//Log_info.LG_found_mouse = TRUE;
	return true;
}

static inline BOOL callmouse(MOUSE_IO* buffer) {
	//����
	IO_STATUS_BLOCK block;
	if (!(NtDeviceIoControlFile(Log_info.LG_input, 0, 0, 0, &block, 0x2a2010, buffer, sizeof(MOUSE_IO), 0, 0) == 0L)) {
		return false;
	}
	return true;
}

static inline bool Init_Drive() {

	if (Log_info.LG_input == 0) {
		//��ʼ��1
		wchar_t buffer0[] = L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
		if (Get_NT_status(buffer0)) {
			return true;
		}
		//��ʼ��2
		wchar_t buffer1[] = L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
		if (Get_NT_status(buffer1)) {
			return true;
		}
	}
	if (Log_info.LG_status == 0) {
		return false;
	}
	return true;
}

static bool LG_Close() {
	//�ͷ�����
	if (Log_info.LG_input != NULL) {
		NtClose(Log_info.LG_input);
		Log_info.LG_input = 0;
	}
	return true;
}

static inline bool Init_IO() {
	//��ʼ��IO
	Log_info.io.unk1 = 0;
	Log_info.io.button = 0;
	Log_info.io.x = 0;
	Log_info.io.y = 0;
	Log_info.io.wheel = 0;
	return true;
}

static inline bool Send_drive(char x, char y) {
	//�����ƶ�������
	Log_info.io.x = x;
	Log_info.io.y = y;

	if (!callmouse(&Log_info.io)) {
		return false;
	}
	return true;
}

static bool LG_move(int x, int y) {

	if (abs(x) > 127 || abs(y) > 127) {
		int x_left = x; int y_left = y;

		if (abs(x) > 127) {
			Send_drive(int(x / abs(x)) * 127, 0);
			x_left = x - int(x / abs(x)) * 127;
		}
		else {
			Send_drive(int(x), 0);
			x_left = 0;
		}

		if (abs(y) > 127) {
			Send_drive(0, int(y / abs(y)) * 127);
			y_left = y - int(y / abs(y)) * 127;
		}
		else {
			Send_drive(0, int(y));
			y_left = 0;
		}

		return LG_move(x_left, y_left);
	}
	else {
		Send_drive(x, y);
	}

	Send_drive(x, y);
	return true;
}

static inline bool Log_test() {
	//�ƶ�����
	MOUSE_IO io{};
	io.unk1 = 0;
	io.button = 0;
	io.x = 10;
	io.y = 10;
	io.wheel = 0;
	if (!callmouse(&io)) {
		std::cout << "Log_test: ��ʧ��,���GHUB(Ҫ��<=2021)" << std::endl;
		return false;
	}
	return true;
}

// -----------------------------------   kmbox�ƶ�  -----------------------------------
//���ò�����
static inline bool Set_baudrate() {
	//��д��������С
	SetupComm(Km_info.com_hwnd, 1024, 1024);

	//��д��ʱ
	COMMTIMEOUTS Out_Time{};
	Out_Time.ReadIntervalTimeout = 1000;		//��ȡ�����ʱ,��λms
	Out_Time.ReadTotalTimeoutConstant = 5000;	//��ȡ�ܳ�ʱ����
	Out_Time.ReadTotalTimeoutMultiplier = 500;	//��ȡ�ܳ�ʱ����
	Out_Time.WriteTotalTimeoutConstant = 2000;	//д���ܳ�ʱ����
	Out_Time.WriteTotalTimeoutMultiplier = 500; //д���ܳ�ʱ����
	SetCommTimeouts(Km_info.com_hwnd, &Out_Time);

	//���ô�����Ϣ
	DCB device_control_info;
	GetCommState(Km_info.com_hwnd, &device_control_info);
	device_control_info.BaudRate = 115200;			//������
	device_control_info.ByteSize = 8;				//8λ����λ
	device_control_info.StopBits = NOPARITY;		//1��ֹͣλ	
	device_control_info.Parity = NOPARITY;			//��żУ��λ
	SetCommState(Km_info.com_hwnd, &device_control_info);

	//���
	PurgeComm(Km_info.com_hwnd, PURGE_TXCLEAR | PURGE_RXCLEAR);

	return true;
}

static inline bool Open_Comx() {

	//��com��
	Km_info.com_hwnd = CreateFileA(
		cfg_info.Move.comx.c_str(),		//���ں�
		GENERIC_READ | GENERIC_WRITE,	//֧�ֶ�д
		0,								//��ռ��ʽ�����ڲ�֧�ֹ���
		NULL,							//��ȫ����
		OPEN_EXISTING,					//�����еĴ����ļ�
		0,								//0��ͬ����ʽ��FILE_FLAG_OVERLAPPED���첽��ʽ
		NULL);							//���ڸ��ƾ����Ĭ��ֵΪNULL���Դ��ڶ��Ըò���������ΪNULL
	if (Km_info.com_hwnd == INVALID_HANDLE_VALUE) {

		std::cout << "Open_Comx: �򿪴���[" << cfg_info.Move.comx.c_str() << "]ʧ�ܣ����Ӳ���ʹ��ں�" << std::endl;
		return false;
	}

	//���ò�����
	if (!Set_baudrate()) {
		std::cout << "Open_Comx: ���ô�����Ϣ(������)ʧ�ܣ����Ӳ��ռ�úͲ�����(115200)" << std::endl;
		return false;
	}
	return true;
}

static inline bool Send_Com(const char* str) {
	//��������
	if (!WriteFile(Km_info.com_hwnd, str, std::strlen(str), 0, NULL)) {
		return false;
	}
	return true;
}

static bool KM_Move(int x, int y) {
	//�����ƶ��ַ���
	std::string km_send;
	km_send = "km.move(" + std::to_string(x) + "," + std::to_string(y) + ")\r\n";
	//����
	if (!Send_Com(km_send.c_str())) {
		std::cout << "KM_Move: �ƶ������ʧ��,�������³�ʼ��kmboxӲ��" << std::endl;
		//���³�ʼ��
		if (!Open_Comx()) {
			return false;
		}
	}
	return true;
}

static bool KM_Close() {
	//���ž��
	if (Km_info.com_hwnd != NULL) {
		if (!CloseHandle(Km_info.com_hwnd)) {
			std::cout << "KM_close: �ͷ�com����ʧ��,������: " << GetLastError() << std::endl;
			return false;
		}
	}
	return true;
}

// -----------------------------------   ������/�׼����ƶ�  -----------------------------------
static bool FOY_Move(int x, int y) {
	//�ƶ�����
	M_MoveR(FOY_info.FOY_hwnd, x, y);
	return true;
}

static bool FOY_Close() {
	//�رսӿ�
	if (FOY_info.FOY_hwnd) {
		M_Close(FOY_info.FOY_hwnd);
	}
	//�ͷž��
	if (FOY_info.FOY_hwnd) {
		if (!CloseHandle(FOY_info.FOY_hwnd)) {
			std::cout << "FOY_Close: �ͷ� ������/�׼��� ʧ��,������:" << GetLastError() << std::endl;
			return false;
		}
	}
	return true;
}

// -----------------------------------   SendInput�ƶ�  -----------------------------------
static inline bool Create_Input() {
	//��������ṹ
	Sen_info.input.type = INPUT_MOUSE;
	Sen_info.input.mi.dx = 0;
	Sen_info.input.mi.dy = 0;
	Sen_info.input.mi.mouseData = 0;
	Sen_info.input.mi.dwFlags = MOUSEEVENTF_MOVE;   //MOUSEEVENTF_ABSOLUTE �������λ��  MOUSEEVENTF_MOVE�����ƶ��¼�
	Sen_info.input.mi.time = 0;
	Sen_info.input.mi.dwExtraInfo = 0;

	return true;
}

static bool Send_Move(int x, int y) {
	//�����ƶ�
	Sen_info.input.mi.dx = x;
	Sen_info.input.mi.dy = y;
	//�ƶ�
	UINT hr = SendInput(1, &Sen_info.input, sizeof(INPUT));
	if (!hr) {
		std::cout << "Send_Move: SendInput����ʧ�ܣ�������: " << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static bool Send_Close() {
	//��
	return true;
}

// -----------------------------------   ��ʼ��  -----------------------------------
static inline bool Init_SendInput() {

	if (!Create_Input()) {
		return false;
	}
	global_data.Move_R = Send_Move;
	global_data.Move_free = Send_Close;
	return true;
}

static inline bool Init_FeiorYi() {
	//�׼���/��������ʼ��
	FOY_info.FOY_hwnd = M_Open(1);
	if (FOY_info.FOY_hwnd == INVALID_HANDLE_VALUE) {
		std::cout << "Init_FeiorYi: ��Ӳ��ʧ��,���Ӳ��" << std::endl;
		return false;
	}
	//������ַ
	global_data.Move_R = FOY_Move;
	global_data.Move_free = FOY_Close;

	return true;
}

static inline bool Init_Serail() {
	//��com��
	if (!Open_Comx()) {
		return false;
	}
	//������ַ
	global_data.Move_R = KM_Move;
	global_data.Move_free = KM_Close;
	return true;
}

static inline bool Init_Logitech() {
	//��ʼ������
	if (!Init_Drive()) {
		std::cout << "Init_Logitech: NT��ʼ��ʧ��" << std::endl;
		return false;
	}
	//��ʼ��IO
	if (!Init_IO()) {
		return false;
	}
	//�����ƶ�
	//if (!Log_test()) {
	//	return false;
	//}

	//������ַ
	global_data.Move_R = LG_move;
	global_data.Move_free = LG_Close;
	return true;
}


bool Init_Move() {
	//��ʼ���ƶ�����
	if (cfg_info.Move.move_manner == 0) {
		if (!Init_Logitech()) {
			std::cout << "Init_Move: �޼���ʼ��ʧ��" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 1) {
		if (!Init_Serail()) {
			std::cout << "Init_Move: KmBox��ʼ��ʧ��" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 2) {
		if (!Init_FeiorYi()) {
			std::cout << "Init_Move: ������/�׼����ʼ��ʧ��" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 3) {
		if (!Init_SendInput()) {
			std::cout << "Init_Move: SendInput��ʼ��ʧ��" << std::endl;
			return false;
		}
	}
	std::cout << "Mouse movement initialization PASS..." << std::endl;
	return true;
}