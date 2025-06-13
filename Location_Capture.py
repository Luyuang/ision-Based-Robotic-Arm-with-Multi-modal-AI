import json
import time
import numpy as np
import cv2
import serial  # 导入 pyserial 库

from Control_Arm import ArmControl
from util.Hand_Eye_Calibration import EyeInHand
from util.Func import Get_EyeInHand_Parameter, Get_One_Camera_Parameter, Get_Two_Camera_Parameter, Json_Updata
from util.Depth_Estimation import Rectify, Count_Disparity, Count_Range_Depth

# 相机标定参数
class CameraParameter():
    def __init__(self):
        self.EIH = EyeInHand()
        self.RT_camera2end = Get_EyeInHand_Parameter("config/calibration_parameter.json")
    
        two_camera_parameter = Get_Two_Camera_Parameter("config/calibration_parameter.json")
        self.left_map1, self.left_map2, self.right_map1, self.right_map2, self.Q = Rectify(two_camera_parameter)
        
        self.mtx, self.dist = Get_One_Camera_Parameter("config/calibration_parameter.json")
        self.fx, self.fy, self.cx, self.cy = self.mtx[0][0], self.mtx[1][1], self.mtx[0][2], self.mtx[1][2]
        
        # 误差微调
        with open('config/argument.json', 'r', encoding='utf-8') as f:
            self.argument = json.load(f)
        self.Z = self.argument["default_Z"]
        self.baseY = self.argument["baseY"]   # 基底坐标系的Y轴的微调
        self.baseX = self.argument["baseX"]   # 基底坐标系的X轴的微调
        
        self.detect_threshold = self.argument["detect_threshold"] # 检测矩形阈值，亮度高则提高
        self.gripper_lenght = self.argument["gripper_lenght"] # 末端到夹爪申直后末端的距离
        self.limit_rectangle = [10, 10, 620, 460] # 限位矩形框 [x, y, width, height]
        
        self.depth = False # 是否开启深度估计
        self.resize = False # 是否开启图像缩放 (缩小一半)

# 计算目标到基地的坐标
def compute_pose(P_target2camera, RT_camera2end, RT_end2base):
    P_end_homogeneous = RT_camera2end @ P_target2camera
    P_base_homogeneous = RT_end2base @ P_end_homogeneous
    x,y,z = P_base_homogeneous[0,0] * 1000, P_base_homogeneous[1,0] * 1000, P_base_homogeneous[2,0] * 1000
    return x,y,z

# 鼠标回调函数
def onmouse_pick_points(event, x, y, flags, pose):
    if event == cv2.EVENT_LBUTTONDOWN: # 左键点击
        if (x > CP.limit_rectangle[0] and x < CP.limit_rectangle[0] + CP.limit_rectangle[2] and y > CP.limit_rectangle[1] and y < CP.limit_rectangle[1] + CP.limit_rectangle[3]):
            if CP.resize:
                x, y = x * 2, y * 2
            # 图像坐标转换为相机坐标
            X = (x - CP.cx) * CP.Z / CP.fx
            Y = (y - CP.cy) * CP.Z / CP.fy
            # 齐次坐标
            P_target2camera = np.array([X, Y, CP.Z, 1]).reshape(4, 1)
            RT_end2base = CP.EIH.pose_to_homogeneous_matrix(pose)
            x, y, z = compute_pose(P_target2camera, CP.RT_camera2end, RT_end2base)
            new_pose = [x + CP.baseX , y + CP.baseY, z + CP.gripper_lenght, 0, 0, -3.14]
            
            AC.Read_Track("list", new_pose, True)
            AC.Run_Arm(start_claw=True)
        
# 矩形位姿获取        
def Rectangle_Pose(box, CP, pose):
    rx, ry, rw, rh = box
    # 计算中心点
    rcx, rcy = rx + rw // 2, ry + rh // 2
    if rcx < 320:
        rcx = rx + rw
    if CP.resize:
        rcx, rcy = rcx * 2, rcy * 2
    # 图像坐标转换为相机坐标
    X = (rcx - CP.cx) * CP.Z / CP.fx
    Y = (rcy - CP.cy) * CP.Z / CP.fy
    # 齐次坐标
    P_target2camera = np.array([X, Y, CP.Z, 1]).reshape(4, 1)
    RT_end2base = CP.EIH.pose_to_homogeneous_matrix(pose)
    x, y, z = compute_pose(P_target2camera, CP.RT_camera2end, RT_end2base)
    new_pose = [x + CP.baseX , y + CP.baseY, z + CP.gripper_lenght, 0, 0, -3.14]
    return new_pose

    
# 检测矩形
def Detect_Rectangle(frame, limit_rectangle, detect_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, detect_threshold, 255,cv2.THRESH_BINARY)
    erosion = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((5,5)))
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓，然后计算其最小外接矩形
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    if max_contour is not None:
        rx, ry, rw, rh = cv2.boundingRect(max_contour)
        x, y, w, h = limit_rectangle
        # 判断rx, ry, rw, rh是否在limit_rectangle范围内
        if rx > x and rx < x + w and ry > y and ry < y + h and rx + rw <= x + w and ry + rh <= y + h:
            return [rx, ry, rw, rh], erosion
        else:
            return None, erosion
    else:
        return None, erosion

# 视频跟腐蚀图拼接
def Video_Merge(frame1, frame2):
    frame = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
    # 将两个图像拼接在一起
    dst = np.hstack((frame1, frame))
    return dst

# 扩展 ArmControl 类，添加串口控制和固定角度设置
class CustomArmControl(ArmControl):
    def __init__(self):
        super().__init__()
        # 初始化夹爪控制的串口，假设夹爪通过 COM10 端口控制
        try:
            self.claw_ser = serial.Serial('COM10', 9600, timeout=1)
            print("夹爪串口初始化成功：COM10")
            time.sleep(2)  # 等待串口初始化完成
        except serial.SerialException as e:
            print(f"夹爪串口初始化失败：{e}")
            self.claw_ser = None

    def set_servo_angle(self, angle):
        """设置舵机角度，范围 0-180 度"""
        if self.claw_ser is None:
            print("夹爪串口未初始化，无法设置舵机角度")
            return
        if 0 <= angle <= 180:
            try:
                self.claw_ser.write(f"f{angle}\n".encode())
                time.sleep(0.1)  # 等待舵机动作完成
                print(f"舵机角度设置为：{angle} 度")
            except serial.SerialException as e:
                print(f"设置舵机角度失败：{e}")
        else:
            print(f"角度 {angle} 超出范围（0-180 度）")

    def Run_Arm(self, start_claw=False):
        """控制机械臂和夹爪，使用固定角度"""
        # 先调用父类的 Run_Arm 方法，确保机械臂移动
        print("Running arm: Moving to target position...")
        super().Run_Arm(start_claw)  # 调用父类的 Run_Arm 方法，执行机械臂移动

        # 控制夹爪
        if start_claw:
            # 夹取：设置舵机角度为 30 度
            self.set_servo_angle(30)
            self.claw_state.value = 1.0  # 更新状态为夹取
            print("Claw closed at 30 degrees")
        else:
            # 释放：设置舵机角度为 0 度
            self.set_servo_angle(0)
            self.claw_state.value = -1.0  # 更新状态为释放
            print("Claw opened at 0 degrees")
            time.sleep(2.0)  # 增加延迟，确保夹爪完全打开

    def __del__(self):
        """在对象销毁时关闭串口"""
        if hasattr(self, 'claw_ser') and self.claw_ser is not None:
            self.claw_ser.close()
            print("夹爪串口已关闭")

if __name__ == "__main__":
    # 使用扩展后的 CustomArmControl 类
    AC = CustomArmControl()
    AC.Set_Arm("COM10")
    AC.Arm_Adjust()
    print("准备中...")
    
    # 更新机械臂的拍照位置
    Json_Updata("config/pose_config.json", "photo_pose", [AC.can_.c_angle.px_out, AC.can_.c_angle.py_out, AC.can_.c_angle.pz_out, AC.can_.c_angle.alpha_out, AC.can_.c_angle.beta_out, AC.can_.c_angle.gama_out])
    
    CP = CameraParameter()
    text = f"Defaultslt Depth: {CP.Z}m"
    if CP.resize:
        CP.limit_rectangle = [int(i / 2) for i in CP.limit_rectangle]
    
    cap = cv2.VideoCapture(0)  # 0是默认摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("Cannot open camera")
    
    # 记录抓取开始时间
    grab_start_time = None
    grab_duration = 5.0  # 抓取后等待 5 秒自动释放

    while True:
        ret, frame = cap.read()
        # 切割为左右两张图片
        frame_L = frame[:, 0:640]
        frame_R = frame[:, 640:1280]
        
        # 纠正畸变
        u, v = frame_R.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CP.mtx, CP.dist, (u, v), 0, (u, v))
        dst1 = cv2.undistort(frame_R, CP.mtx, CP.dist, None, newcameramtx)
        
        # dst1缩小一半
        if CP.resize:
            dst1 = cv2.resize(dst1, (0, 0), fx=0.5, fy=0.5)
            
        # 在dst1上面画一个限位矩形框
        cv2.rectangle(dst1, (CP.limit_rectangle[0], CP.limit_rectangle[1]), (CP.limit_rectangle[0] + CP.limit_rectangle[2], CP.limit_rectangle[1] + CP.limit_rectangle[3]), (0, 255, 0), 2)
        
        # 检测矩形
        box, erosion = Detect_Rectangle(dst1, CP.limit_rectangle, CP.detect_threshold)
                
        # 调用深度计算
        if cv2.waitKey(1) & 0xFF == ord('d'):
            CP.limit_rectangle = [10, 10, 620, 460]
            if CP.resize:
                CP.limit_rectangle = [int(i / 2) for i in CP.limit_rectangle]
            CP.depth = True
            
        # 使用默认深度
        if cv2.waitKey(1) & 0xFF == ord('r'):
            CP.limit_rectangle = [10, 10, 620, 460]
            if CP.resize:
                CP.limit_rectangle = [int(i / 2) for i in CP.limit_rectangle]
            CP.depth = False
            CP.Z = 0.22
            text = f"Defaultslt Depth: {CP.Z}m"
        
        # 计算深度值
        if CP.depth and box is not None:
            disparity = Count_Disparity(frame_L, frame_R, CP.left_map1, CP.right_map2, CP.left_map1, CP.right_map2)
            CP.Z, dis_color = Count_Range_Depth(disparity, box, CP.Q, False, CP.resize)
            text = f"Depth Estimation: {CP.Z:.2f}m"
            if CP.resize:
                dis_color = cv2.resize(dis_color, (0, 0), fx=0.5, fy=0.5)
            dst1 = np.hstack((dst1, dis_color)) 
        
        # 检测矩形驱动
        if box is not None:
            rx, ry, rw, rh = box
            cv2.rectangle(dst1, (rx,ry), (rx + rw, ry + rh), (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                new_pose = Rectangle_Pose(box, CP, now_pose[6:])
                print(f"Target pose: {new_pose}")
                AC.Read_Track("list", new_pose, True)
                AC.Run_Arm(start_claw=True)
                # 记录抓取开始时间
                grab_start_time = time.time()
                print(f"Grab started at {grab_start_time}")
        
        # 自动释放逻辑
        if grab_start_time is not None:
            current_time = time.time()
            print(f"Current time: {current_time}, Grab start time: {grab_start_time}, Time elapsed: {current_time - grab_start_time}")
            if current_time - grab_start_time >= grab_duration:
                print(f"Releasing object: Opening claw, claw_state: {AC.claw_state.value}")
                # 直接通过 Run_Arm 释放（设置角度为 0 度）
                AC.Run_Arm(start_claw=False)
                print(f"After release: claw_state: {AC.claw_state.value}")
                # 重置抓取时间
                grab_start_time = None
        
        # 窗口拼接    
        dst = Video_Merge(dst1, erosion)
        cv2.putText(dst, text, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.imshow('Video', dst)
        
        # 鼠标点击驱动
        now_pose = AC.Get_Pose()
        cv2.setMouseCallback("Video", onmouse_pick_points, now_pose[6:])
        
        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 回到初始位置
            AC.Read_Track("list", AC.pose_config["zero_pose"], False)
            AC.Run_Arm()
            AC.claw_state.value = -1.0
            break
        
    cap.release()
    cv2.destroyAllWindows()