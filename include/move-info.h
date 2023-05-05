#pragma once
#include <winternl.h>

bool Init_Move();


// -----------------------------------   SendInput�ƶ�  -----------------------------------
struct Send_info {
	INPUT input;
};


// -----------------------------------   ������/�׼����ƶ�   -----------------------------------
struct FeiOrYi_info {
	HANDLE FOY_hwnd = NULL;
};

// -----------------------------------   kmbox�ƶ�  -----------------------------------
struct KmBox_info{
	HANDLE com_hwnd = NULL;
};


// -----------------------------------   �޼��ƶ�  -----------------------------------
typedef struct {
	char button;
	char x;
	char y;
	char wheel;
	char unk1;
} MOUSE_IO;

struct Logitech_info {
	MOUSE_IO io{};
	HANDLE LG_input = NULL;
	IO_STATUS_BLOCK LG_io{};
	NTSTATUS LG_status{};
	//BOOL LG_found_mouse;
};
