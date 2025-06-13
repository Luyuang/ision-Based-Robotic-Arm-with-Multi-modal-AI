# import json
# import time
# import os

# from threading import Thread
# import multiprocessing
# import cv2
# import keyboard
# import serial

# from util.Robot_Arm import Can_transfer, claw_control
# from util.Func import Json_Updata

# class ArmControl():
#     def __init__(self):
#         with open('config/argument.json', 'r', encoding='utf-8') as f:
#             self.argument = json.load(f)
#         self.idVendor = int(self.argument["idVendor"],16)
#         self.idProduct = int(self.argument["idProduct"],16)
#         self.can_ = Can_transfer(idVendor=self.idVendor, idProduct=self.idProduct) # 连接USBCan通讯（不同数据线的id可能不一样的，这个需要改！！！）
#         with open('config/pose_config.json', 'r', encoding='utf-8') as f:
#             self.pose_config = json.load(f)
#         self.res = self.pose_config["res"] # 机械臂的睡眠点位置
#         self.zero = self.pose_config["zero"] # 机械臂的安全点位置
#         self.can_.read_angle_flag = True # 读取机械臂的角度标志位
#         self.can_.Start(res=self.res, zero=self.zero) # 初始化及连接机械臂
#         self.targets = [] # 安全点轨迹列表
#         self.sleep_targets = [] # 睡眠点轨迹列表

#         self.print_targets = True # 是否打印轨迹列表

#     # 设置机械臂
#     # def Set_Arm(self, claw_com, res = False, zero = False, enable = False, disable = False, claw_thread = True):
#     #     # 机械臂回到睡眠点标志位
#     #     self.can_.write_res_angle_flag = res
#     #     if self.can_.write_res_angle_flag:
#     #         self.can_._1_edit_angle, self.can_._2_edit_angle, self.can_._3_edit_angle, self.can_._4_edit_angle, self.can_._5_edit_angle, self.can_._6_edit_angle = self.res
#     #     # read_angle_flag
#     #     self.can_.write_zero_angle_flag = zero
#     #     if self.can_.write_zero_angle_flag:
#     #         self.can_._1_edit_angle, self.can_._2_edit_angle, self.can_._3_edit_angle, self.can_._4_edit_angle, self.can_._5_edit_angle, self.can_._6_edit_angle = self.zero
#     #     # 机械臂的电机使能标志位
#     #     self.can_.motor_enable_state = enable
#     #     # 机械臂的电机失能标志位
#     #     self.can_.motor_disable_state = disable
#     #     self.can_.Update()
#     #     time.sleep(0.1)

#     #     #开启夹爪线程
#     #     self.claw_state = multiprocessing.Value('d', 0.0)
#     #     if claw_thread:
#     #         self.clawThread = Thread(target=claw_control, args=(self.claw_state, claw_com)) # 开启机械爪线程
#     #         self.clawThread.start()
    
#     # 更新获取pose_config文件数据
#     def Update_Pose_Data(self):
#         with open('config/pose_config.json', 'r', encoding='utf-8') as f:
#             self.pose_config = json.load(f)
#         self.res = self.pose_config["res"] # 机械臂的睡眠点位置
#         self.zero = self.pose_config["zero"] # 机械臂的安全点位置
        
        
#     # 获取当前位姿
#     def Get_Pose(self):
#         self.can_.Update()
#         pose = [self.can_._1_link_angle,
#                 self.can_._2_link_angle,
#                 self.can_._3_link_angle,
#                 self.can_._4_link_angle,
#                 self.can_._5_link_angle,
#                 self.can_._6_link_angle,
#                 self.can_.c_angle.px_out,
#                 self.can_.c_angle.py_out,
#                 self.can_.c_angle.pz_out,
#                 self.can_.c_angle.alpha_out,
#                 self.can_.c_angle.beta_out,
#                 self.can_.c_angle.gama_out]
#         return pose 
    
#     # 失能校准机械臂
#     def Arm_Adjust(self, enable = False, disable = True):
#         # 机械臂的电机使能标志位
#         self.can_.motor_enable_state = enable
#         # 机械臂的电机失能标志位
#         self.can_.motor_disable_state = disable
#         while True:
#             self.can_.Update()
#             if not self.can_.write_traj_flag:
#                 if self.print_targets:
#                     os.system('cls' if os.name == 'nt' else 'clear')
#                     print('1轴', self.can_._1_link_angle)
#                     print('2轴', self.can_._2_link_angle)
#                     print('3轴', self.can_._3_link_angle)
#                     print('4轴', self.can_._4_link_angle)
#                     print('5轴', self.can_._5_link_angle)
#                     print('6轴', self.can_._6_link_angle)
#                     print('末端x', self.can_.c_angle.px_out)
#                     print('末端y', self.can_.c_angle.py_out)
#                     print('末端z', self.can_.c_angle.pz_out)
#                     print('末端z角度',self. can_.c_angle.alpha_out)
#                     print('末端y角度', self.can_.c_angle.beta_out)
#                     print('末端x角度', self.can_.c_angle.gama_out)
#             if keyboard.is_pressed('q'):
#                 # 机械臂的电机使能标志位
#                 self.can_.motor_enable_state = True
#                 # 机械臂的电机失能标志位
#                 self.can_.motor_disable_state = False
#                 self.can_.Update()
#                 break
#             time.sleep(0.1)  # Delay to reduce CPU usage
    
#     # 录制轨迹    
#     def Transcribe_Track(self, enable = False, disable = True, show_video = False, save_image = False):
#         targets = []
#         # 机械臂的电机使能标志位
#         self.can_.motor_enable_state = enable
#         # 机械臂的电机失能标志位
#         self.can_.motor_disable_state = disable
        
#         # 视频捕获设置
#         if show_video:
#             index = 1
#             cap = cv2.VideoCapture(0)  # 0是默认摄像头\
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             # 检查摄像头是否成功打开
#             if not cap.isOpened():
#                 print("Cannot open camera")
#                 return
            
#         while True:
#             #显示摄像头
#             if show_video:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Can't receive frame (stream end?). Exiting ...")
#                     break
#                 cv2.imshow("R", frame[:,int(640):])
#                 cv2.waitKey(1)
                
#             self.can_.Update()
#             if not self.can_.write_traj_flag:
#                 if self.print_targets:
#                     os.system('cls' if os.name == 'nt' else 'clear')
#                     print('1轴', self.can_._1_link_angle)
#                     print('2轴', self.can_._2_link_angle)
#                     print('3轴', self.can_._3_link_angle)
#                     print('4轴', self.can_._4_link_angle)
#                     print('5轴', self.can_._5_link_angle)
#                     print('6轴', self.can_._6_link_angle)
#                     print('末端x', self.can_.c_angle.px_out)
#                     print('末端y', self.can_.c_angle.py_out)
#                     print('末端z', self.can_.c_angle.pz_out)
#                     print('末端z角度',self. can_.c_angle.alpha_out)
#                     print('末端y角度', self.can_.c_angle.beta_out)
#                     print('末端x角度', self.can_.c_angle.gama_out)
            
#             # Detect 'w' key press
#             if keyboard.is_pressed('w'):
#                 new_target = [
#                     self.can_.c_angle.px_out,
#                     self.can_.c_angle.py_out,
#                     self.can_.c_angle.pz_out,
#                     self.can_.c_angle.alpha_out,
#                     self.can_.c_angle.beta_out,
#                     self.can_.c_angle.gama_out
#                 ]
#                 targets.append(new_target)
#                 print("更新目标点为:", targets)
#                 if show_video and save_image:
#                     cv2.imwrite(f'data/eye_hand_calibration_image/{index}.jpg', frame[:,int(640):])
#                     index += 1
#                     print("保存图片到：", f'data/eye_hand_calibration_image/{index}.jpg')
                
#                 time.sleep(1)  # 延迟显示
#             time.sleep(0.1)  # Delay to reduce CPU usage
#             if keyboard.is_pressed('q'):
#                 self.claw_state.value = -1.0
#                 break
#         # 更新文件进行写入姿态
#         with open('Temp/targets.txt', 'w') as file:
#             # 遍历二维列表的每一行
#             for row in targets:
#                 # 将当前行的元素转换为字符串，并用逗号分隔
#                 row_string = ','.join(map(str, row))
#                 # 写入当前行到文件，并添加换行符
#                 file.write(row_string + '\n')
                
#         print("运动轨迹保存到：", 'Temp/targets.txt')
#         file.close()    
        
#         # 机械臂的电机使能标志位
#         self.can_.motor_enable_state = True
#         # 机械臂的电机失能标志位
#         self.can_.motor_disable_state = False
#         self.can_.Update()  
        
#         if show_video:
#             cap.release()
#             cv2.destroyAllWindows()
     
#     # 校准Pose数据
#     def Calibration_Pose(self):
#         self.can_.Update()
#         # 更新机械臂的初始位置
#         Json_Updata("config/pose_config.json", "zero", [self.can_._1_link_angle, self.can_._2_link_angle, self.can_._3_link_angle, self.can_._4_link_angle, self.can_._5_link_angle, self.can_._6_link_angle])
#         # 更新机械臂的初始的末端位置
#         Json_Updata("config/pose_config.json", "zero_pose", [self.can_.c_angle.px_out, self.can_.c_angle.py_out, self.can_.c_angle.pz_out, self.can_.c_angle.alpha_out, self.can_.c_angle.beta_out, self.can_.c_angle.gama_out])
#         print("初始位置、末端位置 ：更新完成")
              
#     # 读取轨迹        
#     def Read_Track(self, method, file_path, reset = True, command = None):
#         self.targets.clear()
        
#         # 该模式用于测试轨迹任务
#         if method == "txt":
#             # 打开文件进行读取
#             with open(file_path, 'r') as file:
#                 # 逐行读取文件
#                 for line in file:
#                     # 移除行尾的换行符并按逗号分隔字符串
#                     row = line.strip().split(',')
#                     # 将分割后的字符串转换为整数，并添加到二维列表中
#                     self.targets.append([float(item) for item in row])
#                 if reset:
#                     self.targets.append(self.pose_config["zero_pose"])
                    
#         # 该模式用于轨迹任务         
#         elif method == "json":
#             with open(file_path, 'r',  encoding='utf-8') as f:
#                 json_data = json.load(f)
#             self.targets.extend(json_data[command])
#             if reset:
#                 self.targets.append(self.pose_config["zero_pose"])
                
#         # 该模式用于抓取任务
#         elif method == "list":
#             self.targets.append(file_path)
#             if reset:
#                 # 回到拍照位置
#                 self.targets.append(self.pose_config["photo_pose"])
#                 # 放物品
#                 self.targets.append(self.pose_config["place_pose"])
#                 # 回到拍照位置
#                 self.targets.append(self.pose_config["photo_pose"])
                
#     # 单轴运动
#     def Single_Axis(self, axis, angle):
#         self.can_.Update()
#         self.can_._1_edit_angle = self.can_._1_link_angle
#         self.can_._2_edit_angle = self.can_._2_link_angle
#         self.can_._3_edit_angle = self.can_._3_link_angle
#         self.can_._4_edit_angle = self.can_._4_link_angle
#         self.can_._5_edit_angle = self.can_._5_link_angle
#         self.can_._6_edit_angle = self.can_._6_link_angle
#         if (0 < axis and axis <= 6):
#             self.can_.write_angle_flag = True
#             if axis == 1:
#                 self.can_._1_edit_angle = angle
#             elif axis == 2:
#                 self.can_._2_edit_angle = angle
#             elif axis == 3:
#                 self.can_._3_edit_angle = angle
#             elif axis == 4:
#                 self.can_._4_edit_angle = angle
#             elif axis == 5:
#                 self.can_._5_edit_angle = angle
#             elif axis == 6:
#                 self.can_._6_edit_angle = angle
#             self.can_.Update()
#         else:
#             print("输入轴号错误,请输入1~6轴")
            
                        
#     # 执行轨迹
#     def Run_Arm(self, start_claw = False):
#         for i in range(len(self.targets)):
#             self.can_.write_traj_flag = True
#             self.can_.out_traj_button(self.targets[i])
#             t1, t2, t3, t4, t5, t6 = False, False, False, False, False, False
#             ts = time.time()
#             while True:
#                 self.can_.Update()
#                 if not self.can_.write_traj_flag:
#                     if self.print_targets:
#                         os.system('cls' if os.name == 'nt' else 'clear')
#                         print('1轴', self.can_._1_link_angle)
#                         print('2轴', self.can_._2_link_angle)
#                         print('3轴', self.can_._3_link_angle)
#                         print('4轴', self.can_._4_link_angle)
#                         print('5轴', self.can_._5_link_angle)
#                         print('6轴', self.can_._6_link_angle)
#                         print('末端x', self.can_.c_angle.px_out)
#                         print('末端y', self.can_.c_angle.py_out)
#                         print('末端z', self.can_.c_angle.pz_out)
#                         print('末端z角度', self.can_.c_angle.alpha_out)
#                         print('末端y角度', self.can_.c_angle.beta_out)
#                         print('末端x角度', self.can_.c_angle.gama_out)
#                     if abs(self.can_.c_angle.px_out - self.targets[i][0]) < 0.01:
#                         t1 = True
#                     if abs(self.can_.c_angle.py_out - self.targets[i][1]) < 0.01:
#                         t2 = True
#                     if abs(self.can_.c_angle.pz_out - self.targets[i][2]) < 0.01:
#                         t3 = True
#                     if abs(self.can_.c_angle.alpha_out - self.targets[i][3]) < 0.01:
#                         t4 = True
#                     if abs(self.can_.c_angle.beta_out - self.targets[i][4]) < 0.01:
#                         t5 = True
#                     if abs(self.can_.c_angle.gama_out - self.targets[i][5]) < 0.01:
#                         t6 = True
#                 if (t1 and t2 and t3 and t4 and t5 and t6) or time.time() - ts > 2:
#                     break
#             if(start_claw):        
#                 if i == 0:
#                     time.sleep(2)
#                     self.claw_state.value = 1.0
#                     time.sleep(2)
#                 if i == 2:
#                     time.sleep(2)
#                     self.claw_state.value = 0.0
#                     time.sleep(2)
        
# if __name__ == "__main__":
#     AC = ArmControl()
#     # AC.Set_Arm("COM10", claw_thread=True)
#     ser = serial.Serial('COM10', 9600)
    
# ########################################################################################################################################################
# #                                                                    机械臂校准                                                                        #
# ########################################################################################################################################################
#     # AC.Arm_Adjust() # 按q退出，保存校准数据
#     # AC.Calibration_Pose()
    
# ########################################################################################################################################################
# #                                                                    基础动作                                                                          #
# ########################################################################################################################################################

#     # 指令映射表
#     command_actions = {
#         "q": lambda: setattr(AC.claw_state, 'value', -1.0),
#         "4": lambda: ser.write("40\n".encode()),
#         "0": lambda: ser.write("0\n".encode())
#     }

#     while True:
#         try:
#             x = input("输入动作(q退出/4夹紧/0松开/其他执行动作): ").strip()
            
#             # 处理已知指令
#             if x in command_actions:
#                 command_actions[x]()
#                 time.sleep(0.5)
#                 if x == "q":
#                     break
#                 continue
                
#             # 处理特殊动作指令
#             if x in ["组合", "3", "2", "智能抓取", "1"]:
#                 reset = False
#                 AC.Read_Track("json", 'config/motion_config.json', reset, command=x)
#                 AC.Run_Arm()
#             else:
#                 print("未知指令")
                
#         except Exception as e:
#             print(f"发生错误: {e}")

#     # 清理资源
#     ser.close()
#     print("程序结束")
        

# #######################################################################################################################################################
#                                                                     录制轨迹                                                                         #
# #######################################################################################################################################################
#     AC.Transcribe_Track(show_video=True)
    
    
# #######################################################################################################################################################
#                                                                     单轴控制                                                                         #
# #######################################################################################################################################################
#     AC.Single_Axis(1, 50000)
    
    
    
import json
import time
import os

from threading import Thread
import multiprocessing
import cv2
import keyboard

from util.Robot_Arm import Can_transfer, claw_control
from util.Func import Json_Updata

class ArmControl():
    def __init__(self):
        with open('config/argument.json', 'r', encoding='utf-8') as f:
            self.argument = json.load(f)
        self.idVendor = int(self.argument["idVendor"],16)
        self.idProduct = int(self.argument["idProduct"],16)
        self.can_ = Can_transfer(idVendor=self.idVendor, idProduct=self.idProduct) # 连接USBCan通讯（不同数据线的id可能不一样的，这个需要改！！！）
        with open('config/pose_config.json', 'r', encoding='utf-8') as f:
            self.pose_config = json.load(f)
        self.res = self.pose_config["res"] # 机械臂的睡眠点位置
        self.zero = self.pose_config["zero"] # 机械臂的安全点位置
        self.can_.read_angle_flag = True # 读取机械臂的角度标志位
        self.can_.Start(res=self.res, zero=self.zero) # 初始化及连接机械臂
        self.targets = [] # 安全点轨迹列表
        self.sleep_targets = [] # 睡眠点轨迹列表
        self.claw_state = None # 机械爪状态
        self.print_targets = True # 是否打印轨迹列表

    # 设置机械臂
    def Set_Arm(self, claw_com, res = False, zero = False, enable = False, disable = False, claw_thread = True):
        # 机械臂回到睡眠点标志位
        self.can_.write_res_angle_flag = res
        if self.can_.write_res_angle_flag:
            self.can_._1_edit_angle, self.can_._2_edit_angle, self.can_._3_edit_angle, self.can_._4_edit_angle, self.can_._5_edit_angle, self.can_._6_edit_angle = self.res
        # read_angle_flag
        self.can_.write_zero_angle_flag = zero
        if self.can_.write_zero_angle_flag:
            self.can_._1_edit_angle, self.can_._2_edit_angle, self.can_._3_edit_angle, self.can_._4_edit_angle, self.can_._5_edit_angle, self.can_._6_edit_angle = self.zero
        # 机械臂的电机使能标志位
        self.can_.motor_enable_state = enable
        # 机械臂的电机失能标志位
        self.can_.motor_disable_state = disable
        self.can_.Update()
        time.sleep(0.1)

        #开启夹爪线程
        self.claw_state = multiprocessing.Value('d', 0.0)
        if claw_thread:
            self.clawThread = Thread(target=claw_control, args=(self.claw_state, claw_com)) # 开启机械爪线程
            self.clawThread.start()
    
    # 更新获取pose_config文件数据
    def Update_Pose_Data(self):
        with open('config/pose_config.json', 'r', encoding='utf-8') as f:
            self.pose_config = json.load(f)
        self.res = self.pose_config["res"] # 机械臂的睡眠点位置
        self.zero = self.pose_config["zero"] # 机械臂的安全点位置
        
        
    # 获取当前位姿
    def Get_Pose(self):
        self.can_.Update()
        pose = [self.can_._1_link_angle,
                self.can_._2_link_angle,
                self.can_._3_link_angle,
                self.can_._4_link_angle,
                self.can_._5_link_angle,
                self.can_._6_link_angle,
                self.can_.c_angle.px_out,
                self.can_.c_angle.py_out,
                self.can_.c_angle.pz_out,
                self.can_.c_angle.alpha_out,
                self.can_.c_angle.beta_out,
                self.can_.c_angle.gama_out]
        return pose 
    
    # 失能校准机械臂
    def Arm_Adjust(self, enable = False, disable = True):
        # 机械臂的电机使能标志位
        self.can_.motor_enable_state = enable
        # 机械臂的电机失能标志位
        self.can_.motor_disable_state = disable
        while True:
            self.can_.Update()
            if not self.can_.write_traj_flag:
                if self.print_targets:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print('1轴', self.can_._1_link_angle)
                    print('2轴', self.can_._2_link_angle)
                    print('3轴', self.can_._3_link_angle)
                    print('4轴', self.can_._4_link_angle)
                    print('5轴', self.can_._5_link_angle)
                    print('6轴', self.can_._6_link_angle)
                    print('末端x', self.can_.c_angle.px_out)
                    print('末端y', self.can_.c_angle.py_out)
                    print('末端z', self.can_.c_angle.pz_out)
                    print('末端z角度',self. can_.c_angle.alpha_out)
                    print('末端y角度', self.can_.c_angle.beta_out)
                    print('末端x角度', self.can_.c_angle.gama_out)
            if keyboard.is_pressed('q'):
                # 机械臂的电机使能标志位
                self.can_.motor_enable_state = True
                # 机械臂的电机失能标志位
                self.can_.motor_disable_state = False
                self.can_.Update()
                break
            time.sleep(0.1)  # Delay to reduce CPU usage
    
    # 录制轨迹    
    def Transcribe_Track(self, enable = False, disable = True, show_video = False, save_image = False):
        targets = []
        # 机械臂的电机使能标志位
        self.can_.motor_enable_state = enable
        # 机械臂的电机失能标志位
        self.can_.motor_disable_state = disable
        
        # 视频捕获设置
        if show_video:
            index = 1
            cap = cv2.VideoCapture(0)  # 0是默认摄像头\
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # 检查摄像头是否成功打开
            if not cap.isOpened():
                print("Cannot open camera")
                return
            
        while True:
            #显示摄像头
            if show_video:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                cv2.imshow("R", frame[:,int(640):])
                cv2.waitKey(1)
                
            self.can_.Update()
            if not self.can_.write_traj_flag:
                if self.print_targets:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print('1轴', self.can_._1_link_angle)
                    print('2轴', self.can_._2_link_angle)
                    print('3轴', self.can_._3_link_angle)
                    print('4轴', self.can_._4_link_angle)
                    print('5轴', self.can_._5_link_angle)
                    print('6轴', self.can_._6_link_angle)
                    print('末端x', self.can_.c_angle.px_out)
                    print('末端y', self.can_.c_angle.py_out)
                    print('末端z', self.can_.c_angle.pz_out)
                    print('末端z角度',self. can_.c_angle.alpha_out)
                    print('末端y角度', self.can_.c_angle.beta_out)
                    print('末端x角度', self.can_.c_angle.gama_out)
            
            # Detect 'w' key press
            if keyboard.is_pressed('w'):
                new_target = [
                    self.can_.c_angle.px_out,
                    self.can_.c_angle.py_out,
                    self.can_.c_angle.pz_out,
                    self.can_.c_angle.alpha_out,
                    self.can_.c_angle.beta_out,
                    self.can_.c_angle.gama_out
                ]
                targets.append(new_target)
                print("更新目标点为:", targets)
                if show_video and save_image:
                    cv2.imwrite(f'data/eye_hand_calibration_image/{index}.jpg', frame[:,int(640):])
                    index += 1
                    print("保存图片到：", f'data/eye_hand_calibration_image/{index}.jpg')
                
                time.sleep(1)  # 延迟显示
            time.sleep(0.1)  # Delay to reduce CPU usage
            if keyboard.is_pressed('q'):
                self.claw_state.value = -1.0
                break
        # 更新文件进行写入姿态
        with open('Temp/targets.txt', 'w') as file:
            # 遍历二维列表的每一行
            for row in targets:
                # 将当前行的元素转换为字符串，并用逗号分隔
                row_string = ','.join(map(str, row))
                # 写入当前行到文件，并添加换行符
                file.write(row_string + '\n')
                
        print("运动轨迹保存到：", 'Temp/targets.txt')
        file.close()    
        
        # 机械臂的电机使能标志位
        self.can_.motor_enable_state = True
        # 机械臂的电机失能标志位
        self.can_.motor_disable_state = False
        self.can_.Update()  
        
        if show_video:
            cap.release()
            cv2.destroyAllWindows()
     
    # 校准Pose数据
    def Calibration_Pose(self):
        self.can_.Update()
        # 更新机械臂的初始位置
        Json_Updata("config/pose_config.json", "zero", [self.can_._1_link_angle, self.can_._2_link_angle, self.can_._3_link_angle, self.can_._4_link_angle, self.can_._5_link_angle, self.can_._6_link_angle])
        # 更新机械臂的初始的末端位置
        Json_Updata("config/pose_config.json", "zero_pose", [self.can_.c_angle.px_out, self.can_.c_angle.py_out, self.can_.c_angle.pz_out, self.can_.c_angle.alpha_out, self.can_.c_angle.beta_out, self.can_.c_angle.gama_out])
        print("初始位置、末端位置 ：更新完成")
              
    # 读取轨迹        
    def Read_Track(self, method, file_path, reset = True, command = None):
        self.targets.clear()
        
        # 该模式用于测试轨迹任务
        if method == "txt":
            # 打开文件进行读取
            with open(file_path, 'r') as file:
                # 逐行读取文件
                for line in file:
                    # 移除行尾的换行符并按逗号分隔字符串
                    row = line.strip().split(',')
                    # 将分割后的字符串转换为整数，并添加到二维列表中
                    self.targets.append([float(item) for item in row])
                if reset:
                    self.targets.append(self.pose_config["zero_pose"])
                    
        # 该模式用于轨迹任务         
        elif method == "json":
            with open(file_path, 'r',  encoding='utf-8') as f:
                json_data = json.load(f)
            self.targets.extend(json_data[command])
            if reset:
                self.targets.append(self.pose_config["zero_pose"])
                
        # 该模式用于抓取任务
        elif method == "list":
            self.targets.append(file_path)
            if reset:
                # 回到拍照位置
                self.targets.append(self.pose_config["photo_pose"])
                # 放物品
                self.targets.append(self.pose_config["place_pose"])
                # 回到拍照位置
                self.targets.append(self.pose_config["photo_pose"])
                
    # 单轴运动
    def Single_Axis(self, axis, angle):
        self.can_.Update()
        self.can_._1_edit_angle = self.can_._1_link_angle
        self.can_._2_edit_angle = self.can_._2_link_angle
        self.can_._3_edit_angle = self.can_._3_link_angle
        self.can_._4_edit_angle = self.can_._4_link_angle
        self.can_._5_edit_angle = self.can_._5_link_angle
        self.can_._6_edit_angle = self.can_._6_link_angle
        if (0 < axis and axis <= 6):
            self.can_.write_angle_flag = True
            if axis == 1:
                self.can_._1_edit_angle = angle
            elif axis == 2:
                self.can_._2_edit_angle = angle
            elif axis == 3:
                self.can_._3_edit_angle = angle
            elif axis == 4:
                self.can_._4_edit_angle = angle
            elif axis == 5:
                self.can_._5_edit_angle = angle
            elif axis == 6:
                self.can_._6_edit_angle = angle
            self.can_.Update()
        else:
            print("输入轴号错误,请输入1~6轴")
            
                        
    # 执行轨迹
    def Run_Arm(self, start_claw = False):
        for i in range(len(self.targets)):
            self.can_.write_traj_flag = True
            self.can_.out_traj_button(self.targets[i])
            t1, t2, t3, t4, t5, t6 = False, False, False, False, False, False
            ts = time.time()
            while True:
                self.can_.Update()
                if not self.can_.write_traj_flag:
                    if self.print_targets:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print('1轴', self.can_._1_link_angle)
                        print('2轴', self.can_._2_link_angle)
                        print('3轴', self.can_._3_link_angle)
                        print('4轴', self.can_._4_link_angle)
                        print('5轴', self.can_._5_link_angle)
                        print('6轴', self.can_._6_link_angle)
                        print('末端x', self.can_.c_angle.px_out)
                        print('末端y', self.can_.c_angle.py_out)
                        print('末端z', self.can_.c_angle.pz_out)
                        print('末端z角度', self.can_.c_angle.alpha_out)
                        print('末端y角度', self.can_.c_angle.beta_out)
                        print('末端x角度', self.can_.c_angle.gama_out)
                    if abs(self.can_.c_angle.px_out - self.targets[i][0]) < 0.01:
                        t1 = True
                    if abs(self.can_.c_angle.py_out - self.targets[i][1]) < 0.01:
                        t2 = True
                    if abs(self.can_.c_angle.pz_out - self.targets[i][2]) < 0.01:
                        t3 = True
                    if abs(self.can_.c_angle.alpha_out - self.targets[i][3]) < 0.01:
                        t4 = True
                    if abs(self.can_.c_angle.beta_out - self.targets[i][4]) < 0.01:
                        t5 = True
                    if abs(self.can_.c_angle.gama_out - self.targets[i][5]) < 0.01:
                        t6 = True
                if (t1 and t2 and t3 and t4 and t5 and t6) or time.time() - ts > 2:
                    break
            if(start_claw):        
                if i == 0:
                    time.sleep(2)
                    self.claw_state.value = 1.0
                    time.sleep(2)
                if i == 2:
                    time.sleep(2)
                    self.claw_state.value = 0.0
                    time.sleep(2)
        
if __name__ == "__main__":
    AC = ArmControl()
    AC.Set_Arm("COM10", claw_thread=True)
    
    
########################################################################################################################################################
#                                                                    机械臂校准                                                                        #
########################################################################################################################################################
    # AC.Arm_Adjust() # 按q退出，保存校准数据
    # AC.Calibration_Pose()
    
########################################################################################################################################################
#                                                                    基础动作                                                                          #
########################################################################################################################################################
    # while True:
    #     reset = True
    #     x = input("输入动作: ")
    #     try:
    #         if(x == "q"):
    #             AC.claw_state.value = -1.0
    #             break
    #         if x == "立正" or x == "右" or x == "左":
    #             reset = False
    #         AC.Read_Track("json", 'config/motion_config.json', reset, command=x)
    #         AC.Run_Arm()
    #     except:
    #         print("动作有误，请重新输入")
        

########################################################################################################################################################
#                                                                     录制轨迹                                                                         #
########################################################################################################################################################
    # AC.Transcribe_Track(show_video=True)
    
    
########################################################################################################################################################
#                                                                     单轴控制                                                                         #
########################################################################################################################################################
    # AC.Single_Axis(1, 50000)
    
    
    
