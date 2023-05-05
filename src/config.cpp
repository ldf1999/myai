#include <iostream>
#include <string>
#include <direct.h>
#include <windows.h>
#include <fstream>

#include"sf-info.h"
#include"config-info.h"

static bool Parameter_Error = FALSE;		//���������ʶ
static bool Print_Parameter_Debug = FALSE;	//Debug��ӡ�������ƿ��� FALSE / TRUE


// ----------------------------------  tool  ---------------------------------- //
static bool Is_File(const std::string& file_path) {
	std::ifstream file(file_path.c_str());
	return file.good();
}


// ----------------------------------  Check_ini_path  ---------------------------------- //
static inline bool Check_ini() {
	if (!Is_File("config.ini")) {
		return false;
	}
	return true;
}

// ----------------------------------  ����·��  ---------------------------------- //
static inline bool Set_Config_Path() {
	//��ȡ��ǰ����·��
	std::string Root_Path = getcwd(NULL, 0);
	std::string Ini_Name = "\\config.ini";
	cfg_info.Ini_Path = Root_Path + Ini_Name;
	return true;
}

// ----------------------------------  �ַ�����ת��  ---------------------------------- //

std::wstring UTF8ToUnicode(const char* strSrc)
{
	std::wstring wstrRet;

	if (NULL != strSrc) {
		int len = MultiByteToWideChar(CP_UTF8, 0, strSrc, -1, NULL, 0) * sizeof(WCHAR);
		WCHAR* strDst = new(std::nothrow) WCHAR[len + 1];

		if (NULL != strDst) {
			MultiByteToWideChar(CP_UTF8, 0, strSrc, -1, strDst, len);
			wstrRet = strDst;;
			delete[]strDst;
		}
	}
	return wstrRet;
}

std::string UnicodeToAnsi(const WCHAR* strSrc) {
	std::string strRet;

	if (NULL != strSrc) {
		int len = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
		char* strDst = new(std::nothrow) char[len + 1];

		if (NULL != strDst) {
			WideCharToMultiByte(CP_ACP, 0, strSrc, -1, strDst, len, NULL, NULL);
			strRet = strDst;
			delete[]strDst;
		}
	}
	return strRet;
}

std::string UTF8ToAnsi(const char* strSrc) {
	return UnicodeToAnsi(UTF8ToUnicode(strSrc).c_str());
}

// ----------------------------------  ���/��ӡini����  ---------------------------------- //
//���ش���������
static inline void Print_Error_Parameter(int* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}
	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}
static inline void Print_Error_Parameter(float* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}
	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}
static inline void Print_Error_Parameter(std::string* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}

	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}

// ----------------------------------  ���ؼ��ز���  ---------------------------------- //
//���ؽڵ����
static inline char* Load_Str_Parameter(const char* Key, const char* Val) {
	char temp[MAX_PATH];
	GetPrivateProfileStringA(Key, Val, NULL, temp, MAX_PATH, cfg_info.Ini_Path.c_str());
	return temp;
}

//����Int����
static inline void Load_Node_Parameter(int* info, const char* Key, const char* Val) {
	*info = GetPrivateProfileIntA(Key, Val, NULL, cfg_info.Ini_Path.c_str());
	Print_Error_Parameter(info, Key, Val);
}

//����Float����
static inline void Load_Node_Parameter(float* info, const char* Key, const char* Val) {
	*info = strtod(Load_Str_Parameter(Key, Val), NULL);
	Print_Error_Parameter(info, Key, Val);

}

//����String����
static inline void Load_Node_Parameter(std::string* info, const char* Key, const char* Val) {

	*info = Load_Str_Parameter(Key, Val);
	//std::cout << *info << std::endl;
	Print_Error_Parameter(info, Key, Val);
}

//����ڵ�,��Ҫת�ַ�����
static inline void Load_Node_Parameter_Name(std::string* info, const char* Key, const char* Val) {

	char temp[MAX_PATH];
	GetPrivateProfileStringA(Key, Val, NULL, temp, MAX_PATH, cfg_info.Ini_Path.c_str());
	*info = UTF8ToAnsi(temp);
	//std::cout << *info << std::endl;
	Print_Error_Parameter(info, Key, Val);
}


// ----------------------------------  �ڵ������װ  ---------------------------------- //
static void Load_Check_Parameter(int idx) {
	//���
	Load_Node_Parameter(&cfg_info.Check.checking_Version, cfg_info.Node[idx], cfg_info.Check_val[0]);
}

static void Load_Move_Parameter(int idx) {
	//�ƶ�
	Load_Node_Parameter(&cfg_info.Move.move_manner, cfg_info.Node[idx], cfg_info.Move_val[0]);
	Load_Node_Parameter(&cfg_info.Move.comx, cfg_info.Node[idx], cfg_info.Move_val[1]);
}


static void Load_Fov_Parameter(int idx) {
	//FOV 
	Load_Node_Parameter(&cfg_info.Fov.fov_off, cfg_info.Node[idx], cfg_info.Fov_val[0]);
	Load_Node_Parameter(&cfg_info.Fov.game_HFOV, cfg_info.Node[idx], cfg_info.Fov_val[1]);
	Load_Node_Parameter(&cfg_info.Fov.game_VFOV, cfg_info.Node[idx], cfg_info.Fov_val[2]);
	Load_Node_Parameter(&cfg_info.Fov.game_x_pixel, cfg_info.Node[idx], cfg_info.Fov_val[3]);
	Load_Node_Parameter(&cfg_info.Fov.game_y_pixel, cfg_info.Node[idx], cfg_info.Fov_val[4]);
}

static void Load_Pid_Parameter(int idx) {
	//PID
	Load_Node_Parameter(&cfg_info.Pid.pid_off, cfg_info.Node[idx], cfg_info.Pid_val[0]);
	Load_Node_Parameter(&cfg_info.Pid.kp_x, cfg_info.Node[idx], cfg_info.Pid_val[1]);
	Load_Node_Parameter(&cfg_info.Pid.ki_x, cfg_info.Node[idx], cfg_info.Pid_val[2]);
	Load_Node_Parameter(&cfg_info.Pid.kd_x, cfg_info.Node[idx], cfg_info.Pid_val[3]);
	Load_Node_Parameter(&cfg_info.Pid.kp_y, cfg_info.Node[idx], cfg_info.Pid_val[4]);
	Load_Node_Parameter(&cfg_info.Pid.ki_y, cfg_info.Node[idx], cfg_info.Pid_val[5]);
	Load_Node_Parameter(&cfg_info.Pid.kd_y, cfg_info.Node[idx], cfg_info.Pid_val[6]);
	Load_Node_Parameter(&cfg_info.Pid.sample_num, cfg_info.Node[idx], cfg_info.Pid_val[7]);
}

static void Load_Aim_Parameter(int idx) {
	//Aim
	Load_Node_Parameter(&cfg_info.Aim.range_top, cfg_info.Node[idx], cfg_info.Aim_val[0]);
	Load_Node_Parameter(&cfg_info.Aim.range_bottom, cfg_info.Node[idx], cfg_info.Aim_val[1]);
	Load_Node_Parameter(&cfg_info.Aim.range_left, cfg_info.Node[idx], cfg_info.Aim_val[2]);
	Load_Node_Parameter(&cfg_info.Aim.range_right, cfg_info.Node[idx], cfg_info.Aim_val[3]);
	Load_Node_Parameter(&cfg_info.Aim.cla_off, cfg_info.Node[idx], cfg_info.Aim_val[4]);
	Load_Node_Parameter(&cfg_info.Aim.label_chose, cfg_info.Node[idx], cfg_info.Aim_val[5]);
	Load_Node_Parameter(&cfg_info.Aim.move_offset, cfg_info.Node[idx], cfg_info.Aim_val[6]);
}

static void Load_Fire_Parameter(int idx) {
	//�Զ�����
	Load_Node_Parameter(&cfg_info.Fire.auto_off, cfg_info.Node[idx], cfg_info.Fire_val[0]);
	Load_Node_Parameter(&cfg_info.Fire.auto_method, cfg_info.Node[idx], cfg_info.Fire_val[1]);
	Load_Node_Parameter(&cfg_info.Fire.auto_sleep, cfg_info.Node[idx], cfg_info.Fire_val[2]);
}

static void Load_Key_Parameter(int idx) {
	//����
	Load_Node_Parameter(&cfg_info.Key.aim_off, cfg_info.Node[idx], cfg_info.Key_val[0]);
	Load_Node_Parameter(&cfg_info.Key.key_method, cfg_info.Node[idx], cfg_info.Key_val[1]);
	Load_Node_Parameter(&cfg_info.Key.button_key1, cfg_info.Node[idx], cfg_info.Key_val[2]);
	Load_Node_Parameter(&cfg_info.Key.button_key2, cfg_info.Node[idx], cfg_info.Key_val[3]);
	Load_Node_Parameter(&cfg_info.Key.end_key, cfg_info.Node[idx], cfg_info.Key_val[4]);
}

static void Load_Pred_Parameter(int idx) {
	//����
	Load_Node_Parameter(&cfg_info.Pred.engine_path, cfg_info.Node[idx], cfg_info.Pred_val[0]);
	Load_Node_Parameter(&cfg_info.Pred.frame, cfg_info.Node[idx], cfg_info.Pred_val[1]);
	Load_Node_Parameter(&cfg_info.Pred.conf, cfg_info.Node[idx], cfg_info.Pred_val[2]);
	Load_Node_Parameter(&cfg_info.Pred.iou, cfg_info.Node[idx], cfg_info.Pred_val[3]);
	Load_Node_Parameter(&cfg_info.Pred.sleep, cfg_info.Node[idx], cfg_info.Pred_val[4]);
}

static void Load_Windows_Parameter(int idx) {
	//����
	Load_Node_Parameter(&cfg_info.Windows.capture_method, cfg_info.Node[idx], cfg_info.Win_val[0]);
	Load_Node_Parameter(&cfg_info.Windows.win32_method, cfg_info.Node[idx], cfg_info.Win_val[1]);
	//����ڵ㣬��Ҫ��������
	Load_Node_Parameter_Name(&cfg_info.Windows.win32_name, cfg_info.Node[idx], cfg_info.Win_val[2]);
	Load_Node_Parameter(&cfg_info.Windows.show, cfg_info.Node[idx], cfg_info.Win_val[3]);
}
static void Load_Other_Parameter(int idx) {
	//����
	Load_Node_Parameter(&cfg_info.Other.console_refresh, cfg_info.Node[idx], cfg_info.Other_val[0]);
}


static bool Load_Parameter() {
	//��װ���ؽڵ����
	Load_Check_Parameter(0);
	Load_Move_Parameter(1);
	Load_Fov_Parameter(2);
	Load_Pid_Parameter(3);
	Load_Aim_Parameter(4);
	Load_Fire_Parameter(5);
	Load_Key_Parameter(6);
	Load_Pred_Parameter(7);
	Load_Windows_Parameter(8);
	Load_Other_Parameter(9);

	return true;
}
// ----------------------------------  Init  ---------------------------------- //
//����ini����
static bool Init_Parameter() {
	//����ini����
	if (!Load_Parameter()) {
		return false;
	}
	//��֤engine�ļ��Ƿ����
	if (!Is_File(cfg_info.Pred.engine_path)) {
		std::cout << "�Ҳ���[ " << cfg_info.Pred.engine_path << " ]�ļ�,����ļ��Ƿ���ں�ini·�������Ƿ���ȷ(�ϸ���)" << std::endl;
		system("pause");
		return false;
	}
	return true;
}


bool Init_config() {
	//����Ƿ����ini�ļ�
	if (!Check_ini) {
		std::cout << "Check_Head: config.ini �ļ�������" << std::endl;
		return false;
	}

	//����ini�ļ�·��
	if (!Set_Config_Path()) {
		std::cout << "Set_Config_Path: ����config.ini·��ʧ��" << std::endl;
		return false;
	}
	//����ini����
	if (!Init_Parameter()) {
		std::cout << "Init_Parameter: ����ini��������ʧ��" << std::endl;
		return false;
	}

	//�Ƿ���ڶ�ȡ�������
	if (Parameter_Error) {
		std::cout << "Init_Parameter: ��ini������ȡ����,������" << std::endl;
		return false;
	}
	std::cout << "Load config.ini Parameter PASS..." << std::endl;
	return true;
}

// ----------------------------------  Updata Thread  ---------------------------------- //

long old_modify_time = 0;

static inline bool Get_Last_File_Time() {
	//��ȡ����޸�ʱ��
	int fd = 0;
	int size_fp = 0;
	struct stat buff {};

	FILE* fp = fopen(cfg_info.Ini_Path.c_str(), "r");
	if (fp == NULL) {
		std::cout << "���ļ�ʧ��" << std::endl;
		return false;
	}

	//��ȡ�ļ�������
	fd = fileno(fp);
	//��ȡ�ļ�״̬
	fstat(fd, &buff);
	//��ȡ�ļ���С
	size_fp = buff.st_size;
	//��ȡ�ļ��޸�ʱ��
	long modify_time = buff.st_mtime;
	fclose(fp);

	if (old_modify_time != modify_time){
		old_modify_time = modify_time;
		return true;
	}
	return false;
}

//��̬��ȡ����
static inline bool Load_Updata_Parameter() {

	//�̶��ڵ��ȡ
	Load_Fov_Parameter(2);
	Load_Pid_Parameter(3);
	Load_Aim_Parameter(4);
	Load_Fire_Parameter(5);
	Load_Key_Parameter(6);
	Load_Other_Parameter(9);

	//��ɢ�ڵ��ȡ
	//Pred
	Load_Node_Parameter(&cfg_info.Pred.conf, cfg_info.Node[7], cfg_info.Pred_val[2]);
	Load_Node_Parameter(&cfg_info.Pred.iou, cfg_info.Node[7], cfg_info.Pred_val[3]);
	Load_Node_Parameter(&cfg_info.Pred.sleep, cfg_info.Node[7], cfg_info.Pred_val[4]);
	//win
	Load_Node_Parameter(&cfg_info.Windows.show, cfg_info.Node[8], cfg_info.Win_val[3]);

	return true;
}


//��̬��ȡ�߳�
bool dynamic_thread_loop() {

	while (!Thread_Should_Stop()) {
		//��ȡ����޸�ʱ��
		if (Get_Last_File_Time()){
			//���ز���
			Load_Updata_Parameter();
		}
		//��ʱ���
		Sleep(1800);
	}
	std::cout << "��̬��ȡ�߳��˳�" << std::endl;
	return true;
}
