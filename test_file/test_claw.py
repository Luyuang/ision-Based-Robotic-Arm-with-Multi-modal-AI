import cv2

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow('left', frame[:,0:int(frame.shape[1]/2)])
#     cv2.imshow('right', frame[:,int(frame.shape[1]/2):])39

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import serial
import time

# 设置串口连接参数
ser = serial.Serial('COM10', 9600)  # 'COM3'应替换为实际的ESP端口号

def set_servo_angle(angle):
    if 0 <= angle <= 180:
        ser.write(f"{angle}\n".encode())  # 发送角度数据到ESP
        time.sleep(0.1)  # 等待ESP处理

try:
    while True:
        angle = input("输入舵机角度 (0-180): ")
        if angle.isdigit():
            angle1 = int(angle)
            set_servo_angle(angle1)
        else:
            print("请输入有效的数字!")
except KeyboardInterrupt:
    print("程序结束")
finally:
    ser.close()