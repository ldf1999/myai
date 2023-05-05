
#include <cmath>

#include "sf-info.h"
#include "aim-info.h"
#include "config-info.h"
#include "process-info.h"
#include <chrono>

#define pi 3.14159265358979323846   

static struct PID pid_x;		
static struct PID pid_y;
int kg;




// ----------------------------- FOV ----------------------------- //
static inline void FOV_X(float& move_x) {
	float x = (global_data.window_width * 0.5f) / tan((cfg_info.Fov.game_HFOV * pi) / 180 * 0.5f);
	move_x = atan(move_x / x) * (cfg_info.Fov.game_x_pixel / (2 * pi));
}

static inline void FOV_Y(float& move_y) {
	float y = (global_data.window_height * 0.5f) / tan((cfg_info.Fov.game_VFOV * pi) / 180 * 0.5f);
	move_y = atan(move_y / y) * (cfg_info.Fov.game_y_pixel / pi);
}

// ----------------------------- PID ----------------------------- //
//����ʽ
static inline void Incremental(PID pid, float& move, const float& kp, const float& ki, const float& kd, const int& num) {
	pid.SetSpeed = move;					//��¼�����ƶ�����
	pid.Error = pid.SetSpeed - pid.ActualSpeed;		//��¼��������һ�ε����
	pid.IncrementSpeed = kp * (pid.Error - pid.Err_Next);// +(ki * pid.Error) + kd * (pid.Error - 2 * pid.Err_Next + pid.Err_last);

	//���׿����ֱ���
	pid.Sample_frame++;
	if (pid.Sample_frame % num == 0) {
		pid.ActualSpeed += pid.IncrementSpeed;	//�����ۻ�
		pid.Sample_frame = 0;
	}

	//pid.ActualSpeed += pid.IncrementSpeed;	//�����ۻ�
	pid.Err_last = pid.Err_Next;			//
	pid.Err_Next = pid.Error;				//��ǰ���תΪ��һ�εĲ���
	move = pid.IncrementSpeed;
}

//x��
static inline void PID_X(float& move_x) {
	Incremental(pid_x, move_x, cfg_info.Pid.kp_x, cfg_info.Pid.ki_x, cfg_info.Pid.kd_x, cfg_info.Pid.sample_num);
}

//y��
static inline void PID_Y(float& move_y) {
	Incremental(pid_y, move_y, cfg_info.Pid.kp_y, cfg_info.Pid.ki_y, cfg_info.Pid.kd_y, cfg_info.Pid.sample_num);
}

//ƫ��
static inline void Offset(int &y,int &width) {


	if ((0.f< cfg_info.Aim.move_offset) && (cfg_info.Aim.move_offset <= 1.f)) {
		y = y - (width * cfg_info.Aim.move_offset);
		return;
	}
	
	//��λ
	if (cfg_info.Aim.move_offset>1.f)
		y = y - (width * 1.f);
	else 
		y = y - (width * 0.1f);
}


// ----------------------------- �����֧ ----------------------------- //
//���鷶Χ
static inline bool Check_Range(float& move_x, float& move_y) {
	if (cfg_info.Aim.range_top < abs(move_y) || cfg_info.Aim.range_bottom < abs(move_y) ||
		cfg_info.Aim.range_left < abs(move_x) || cfg_info.Aim.range_right < abs(move_x)
		) {

		return false;
	}
	return true;
}

static inline bool FOV_PID(float& move_x, float& move_y) {

	//FOV����
	if (cfg_info.Fov.fov_off != 0) {
		FOV_X(move_x);
		FOV_Y(move_y);
	}

	//PID����
	if (cfg_info.Pid.pid_off != 0) {
		PID_X(move_x);
		PID_Y(move_y);
	}

	return true;
}

// ----------------------------- �Զ���� ----------------------------- //
//�Զ��������
static inline void Key_click() {

	//����
	INPUT input{};
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;   //MOUSEEVENTF_LEFTDOWN �������
	input.mi.time = 0;
	input.mi.dwExtraInfo = 0;
	SendInput(1, &input, sizeof(INPUT));

	//�ɿ�
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTUP;   // MOUSEEVENTF_LEFTUP  ����ɿ�
	input.mi.time = 0;
	input.mi.dwExtraInfo = 0;

	SendInput(1, &input, sizeof(INPUT));
}

//�����ʱ
std::chrono::system_clock::time_point Sleep_Start = std::chrono::system_clock::now();
static inline bool Auto_Sleep() {
	std::chrono::system_clock::time_point Sleep_End = std::chrono::system_clock::now();
	return std::chrono::duration_cast <std::chrono::milliseconds> (Sleep_End - Sleep_Start).count() >= cfg_info.Fire.auto_sleep;
}

//�Զ����
static inline void Auto_fire_func(float* move_x, float* move_y) {
	//�Զ������ʽ - �ƶ���ǹ
	if (cfg_info.Fire.auto_method  == 0) {
		global_data.Move_R(int(*move_x), int(*move_y));
		if (Auto_Sleep()) {
			Key_click();
			Sleep_Start = std::chrono::system_clock::now();
		}
	}

	if (cfg_info.Fire.auto_method  == 1) {
		//��Χ��ǹ
		if ((Process_data.target_xywh.width * 0.5f) > abs(*move_x) && (Process_data.target_xywh.height * 0.5f) > abs(*move_y) && Auto_Sleep()) {
			Key_click();
			Sleep_Start = std::chrono::system_clock::now();
		}
	}
}
// ----------------------------- �ƶ��߼� ----------------------------- //
//�ƶ��߼�
static inline bool Move_logic(float* move_x, float* move_y) {

	//���鷽ʽ
	if (cfg_info.Key.key_method == 0) {
		//�����ƶ�
		if (GetAsyncKeyState(cfg_info.Key.button_key1) || GetAsyncKeyState(cfg_info.Key.button_key2)) {
			//�Զ����
			if (cfg_info.Fire.auto_off == 1) {
				Auto_fire_func(move_x, move_y);

			}
			else {
				global_data.Move_R(int(*move_x), int(*move_y));
			}
		}
	}
	else {
		if (cfg_info.Fire.auto_off == 1) {
			Auto_fire_func(move_x, move_y);
		}
		else {
			global_data.Move_R(int(*move_x), int(*move_y));
		}
	}

	return true;
}

// ----------------------------- aim Loop ----------------------------- //
DWORD WINAPI aim_func() {


	while (true) {
		//�ȴ������ź�
		WaitForSingleObject(Aim_Event, INFINITE);
		//�����Ƿ��˳�
		if (Thread_Should_Stop()) {
			break;
		}
		//���
		Process_data.recent.clear();
		if (cfg_info.Aim.cla_off) {


			for (int i = 0; i < Process_data.indices.size(); i++) {
				if (Process_data.Classes[Process_data.indices[i]] == cfg_info.Aim.label_chose) {
					for (int i = 0; i < Process_data.indices.size(); i++) {


						Process_data.recent.push_back(
							pow((global_data.origin_x + Process_data.boxes[Process_data.indices[i]].x) - global_data.cx, 2) +
							pow((global_data.origin_y + Process_data.boxes[Process_data.indices[i]].y) - global_data.cy, 2));


					}

					//���Ŀ�������
					int idx = Process_data.indices[
						std::distance(std::begin(Process_data.recent),

							std::min_element(std::begin(Process_data.recent), std::end(Process_data.recent))//min_element()
						)	//distance()
					];	//Process_data.indices[]

					//���Ŀ���x	(��ͼƬ����)
					Process_data.target_xywh.x = Process_data.boxes[idx].x;
					//���Ŀ���y	(��ͼƬ����)
					Process_data.target_xywh.y = Process_data.boxes[idx].y;
					//���Ŀ���width	(��ͼƬ����)
					Process_data.target_xywh.width = Process_data.boxes[idx].width;
					//���Ŀ���height	(��ͼƬ����)
					Process_data.target_xywh.height = Process_data.boxes[idx].height;

					//Debug ����
					//std::cout
					//	<< "x: " << Process_data.target_xywh.x
					//	<< "\ty: " << Process_data.target_xywh.y
					//	<< "\tweidth: " << Process_data.target_xywh.width
					//	<< "\theight: " << Process_data.target_xywh.height
					//	<< std::endl;

					//ƫ�Ƽ���
					if (cfg_info.Aim.move_offset != 0) {
						Offset(Process_data.target_xywh.y, Process_data.target_xywh.width);
					}

					// ��ͼ��ԭ��(global_data.origin_x) + Ŀ�����ĵ�(Process_data.target_xywh.x) = Ŀ������Ļ�����ĵ� 
					// Ŀ������Ļ�����ĵ� - �������ĵ�(global_data.cx) = ׼�Ǻ�Ŀ��֮��ľ���(move_x)
					float move_x = global_data.origin_x + Process_data.target_xywh.x - global_data.cx;
					float move_y = global_data.origin_y + Process_data.target_xywh.y - global_data.cy;

					//����Ƿ������鷶Χ��
					if (!Check_Range(move_x, move_y)) {
						continue;
					}

					//FOV + PID
					if (!FOV_PID(move_x, move_y)) {
						std::cout << "fov + pid�������δ֪�������������ƶ�" << std::endl;
						continue;
					}

					//�ƶ��߼�
					if (!Move_logic(&move_x, &move_y)) {
						std::cout << "�ƶ��߼�����δ֪�������������ƶ�" << std::endl;
						continue;
					}

					
				}
				
			}
		}
		else {


			//��������Ŀ���ŷʽ����
			for (int i = 0; i < Process_data.indices.size(); i++) {


				Process_data.recent.push_back(
					pow((global_data.origin_x + Process_data.boxes[Process_data.indices[i]].x) - global_data.cx, 2) +
					pow((global_data.origin_y + Process_data.boxes[Process_data.indices[i]].y) - global_data.cy, 2));


			}

			//���Ŀ�������
			int idx = Process_data.indices[
				std::distance(std::begin(Process_data.recent),

					std::min_element(std::begin(Process_data.recent), std::end(Process_data.recent))//min_element()
				)	//distance()
			];	//Process_data.indices[]

			//���Ŀ���x	(��ͼƬ����)
			Process_data.target_xywh.x = Process_data.boxes[idx].x;
			//���Ŀ���y	(��ͼƬ����)
			Process_data.target_xywh.y = Process_data.boxes[idx].y;
			//���Ŀ���width	(��ͼƬ����)
			Process_data.target_xywh.width = Process_data.boxes[idx].width;
			//���Ŀ���height	(��ͼƬ����)
			Process_data.target_xywh.height = Process_data.boxes[idx].height;

			//Debug ����
			//std::cout
			//	<< "x: " << Process_data.target_xywh.x
			//	<< "\ty: " << Process_data.target_xywh.y
			//	<< "\tweidth: " << Process_data.target_xywh.width
			//	<< "\theight: " << Process_data.target_xywh.height
			//	<< std::endl;

			//ƫ�Ƽ���
			if (cfg_info.Aim.move_offset != 0) {
				Offset(Process_data.target_xywh.y, Process_data.target_xywh.width);
			}

			// ��ͼ��ԭ��(global_data.origin_x) + Ŀ�����ĵ�(Process_data.target_xywh.x) = Ŀ������Ļ�����ĵ� 
			// Ŀ������Ļ�����ĵ� - �������ĵ�(global_data.cx) = ׼�Ǻ�Ŀ��֮��ľ���(move_x)
			float move_x = global_data.origin_x + Process_data.target_xywh.x - global_data.cx;
			float move_y = global_data.origin_y + Process_data.target_xywh.y - global_data.cy;

			//����Ƿ������鷶Χ��
			if (!Check_Range(move_x, move_y)) {
				continue;
			}

			//FOV + PID
			if (!FOV_PID(move_x, move_y)) {
				std::cout << "fov + pid�������δ֪�������������ƶ�" << std::endl;
				continue;
			}

			//�ƶ��߼�
			if (!Move_logic(&move_x, &move_y)) {
				std::cout << "�ƶ��߼�����δ֪�������������ƶ�" << std::endl;
				continue;
			}
		}
		
	}
	std::cout << "�����߳��˳�..." << std::endl;
	return true;
}

