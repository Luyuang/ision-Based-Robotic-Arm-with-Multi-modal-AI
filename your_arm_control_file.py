import matplotlib.pyplot as plt
import numpy as np
from Control_Arm import ArmControl  # 替换为你的文件名
import os
import time

def inverse_kinematics(target_pose, scale_factor=1000):
    """
    假设的逆运动学函数，将末端位姿转换为关节角度
    参数:
        target_pose: [px, py, pz, alpha, beta, gama]
        scale_factor: 缩放因子，用于匹配实际角度的范围
    返回:
        关节角度 [angle1, angle2, angle3, angle4, angle5, angle6]
    注意: 这里是简化模拟，实际需要根据机械臂的运动学模型实现
    """
    px, py, pz, alpha, beta, gama = target_pose
    # 假设关节角度与末端位姿有简单的线性关系，并应用缩放因子
    angle1 = px * scale_factor
    angle2 = py * scale_factor
    angle3 = pz * scale_factor
    angle4 = alpha * scale_factor
    angle5 = beta * scale_factor
    angle6 = gama * scale_factor
    return [angle1, angle2, angle3, angle4, angle5, angle6]

def convert_to_degrees(angle, pulses_per_degree=1000):
    """
    将编码器值（脉冲数）转换为度数
    参数:
        angle: 编码器值（脉冲数）
        pulses_per_degree: 每度对应的脉冲数（需要根据机械臂参数调整）
    返回:
        角度（度数）
    """
    return angle / pulses_per_degree

def collect_arm_data(arm_control, pulses_per_degree=1000):
    """从 ArmControl 的 Run_Arm 方法中采集实际和理想角度"""
    actual_angles = {i: [] for i in range(1, 7)}
    ideal_angles = {i: [] for i in range(1, 7)}
    
    for i in range(len(arm_control.targets)):
        arm_control.can_.write_traj_flag = True
        arm_control.can_.out_traj_button(arm_control.targets[i])
        
        # 尝试直接从 Can_transfer 获取理想关节角度
        # 如果 out_traj_button 正确更新了 _1_edit_angle 等变量，可以直接使用
        current_ideal = [
            arm_control.can_._1_edit_angle,
            arm_control.can_._2_edit_angle,
            arm_control.can_._3_edit_angle,
            arm_control.can_._4_edit_angle,
            arm_control.can_._5_edit_angle,
            arm_control.can_._6_edit_angle
        ]
        
        # 如果 _1_edit_angle 等变量不可靠，则使用逆运动学计算
        if all(angle == 0 for angle in current_ideal):  # 如果全为0，说明未正确更新
            current_ideal = inverse_kinematics(arm_control.targets[i], scale_factor=100000)
        
        print(f"轨迹点 {i+1} 的末端位姿: {arm_control.targets[i]}")
        print(f"轨迹点 {i+1} 的理想关节角度: {current_ideal}")
        
        t1, t2, t3, t4, t5, t6 = False, False, False, False, False, False
        ts = time.time()
        while True:
            arm_control.can_.Update()
            # 记录实际角度（转换为度数）
            actual_angles[1].append(convert_to_degrees(arm_control.can_._1_link_angle, pulses_per_degree))
            actual_angles[2].append(convert_to_degrees(arm_control.can_._2_link_angle, pulses_per_degree))
            actual_angles[3].append(convert_to_degrees(arm_control.can_._3_link_angle, pulses_per_degree))
            actual_angles[4].append(convert_to_degrees(arm_control.can_._4_link_angle, pulses_per_degree))
            actual_angles[5].append(convert_to_degrees(arm_control.can_._5_link_angle, pulses_per_degree))
            actual_angles[6].append(convert_to_degrees(arm_control.can_._6_link_angle, pulses_per_degree))
            # 记录理想角度（转换为度数）
            ideal_angles[1].append(convert_to_degrees(current_ideal[0], pulses_per_degree))
            ideal_angles[2].append(convert_to_degrees(current_ideal[1], pulses_per_degree))
            ideal_angles[3].append(convert_to_degrees(current_ideal[2], pulses_per_degree))
            ideal_angles[4].append(convert_to_degrees(current_ideal[3], pulses_per_degree))
            ideal_angles[5].append(convert_to_degrees(current_ideal[4], pulses_per_degree))
            ideal_angles[6].append(convert_to_degrees(current_ideal[5], pulses_per_degree))
            
            if not arm_control.can_.write_traj_flag:
                if abs(arm_control.can_.c_angle.px_out - arm_control.targets[i][0]) < 0.01:
                    t1 = True
                if abs(arm_control.can_.c_angle.py_out - arm_control.targets[i][1]) < 0.01:
                    t2 = True
                if abs(arm_control.can_.c_angle.pz_out - arm_control.targets[i][2]) < 0.01:
                    t3 = True
                if abs(arm_control.can_.c_angle.alpha_out - arm_control.targets[i][3]) < 0.01:
                    t4 = True
                if abs(arm_control.can_.c_angle.beta_out - arm_control.targets[i][4]) < 0.01:
                    t5 = True
                if abs(arm_control.can_.c_angle.gama_out - arm_control.targets[i][5]) < 0.01:
                    t6 = True
            if (t1 and t2 and t3 and t4 and t5 and t6) or time.time() - ts > 2:
                break
    
    return actual_angles, ideal_angles

def plot_angle_comparison(actual_angles, ideal_angles, output_dir="angle_plots"):
    """绘制实际角度和理想角度的折线图"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for axis in range(1, 7):
        plt.figure(figsize=(10, 6))
        time_steps = np.arange(len(actual_angles[axis]))
        plt.plot(time_steps, actual_angles[axis], label=f'Actual Angle (Axis {axis})', color='blue', linewidth=1.5)
        plt.plot(time_steps, ideal_angles[axis], label=f'Ideal Angle (Axis {axis})', color='red', linestyle='--', linewidth=1.5)
        plt.xlabel('Time Steps')
        plt.ylabel('Angle (degrees)')
        plt.title(f'Angle Comparison for Axis {axis}')
        plt.legend()
        plt.grid(True)
        output_file = os.path.join(output_dir, f'angle_comparison_axis_{axis}.png')
        plt.savefig(output_file)
        plt.close()
        print(f"已保存折线图: {output_file}")

if __name__ == "__main__":
    # 初始化机械臂控制类
    AC = ArmControl()
    AC.Set_Arm("COM10", claw_thread=True)
    
    # 加载一个轨迹
    AC.Read_Track("txt", "Temp/targets.txt", reset=True)
    
    # 打印轨迹数据以调试
    print("加载的轨迹数据 (self.targets):")
    for i, target in enumerate(AC.targets):
        print(f"轨迹点 {i+1}: {target}")
    
    # 采集数据
    print("正在采集机械臂数据...")
    actual_angles, ideal_angles = collect_arm_data(AC, pulses_per_degree=1000)
    
    # 打印部分数据以调试
    print("实际角度 (Axis 1) 前10个值:", actual_angles[1][:10])
    print("理想角度 (Axis 1) 前10个值:", ideal_angles[1][:10])
    
    # 绘制折线图
    print("正在绘制折线图...")
    plot_angle_comparison(actual_angles, ideal_angles)
    print("完成！请查看 'angle_plots' 目录中的折线图。")