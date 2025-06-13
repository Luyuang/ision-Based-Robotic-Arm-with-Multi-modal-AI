import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("CUDA 设备数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(0))


        # try:
    #     ser = serial.Serial('COM10', 9600, timeout=1)
    #     time.sleep(2)  # 等待串口初始化
    # except Exception as e:
    #     print(f"串口初始化失败: {e}")
    #     exit()

    # # 指令映射表
    # command_actions = {
    #     "q": lambda: setattr(AC.claw_state, 'value', -1.0),
    #     "4": lambda: ser.write("40\n".encode()),
    #     "0": lambda: ser.write("0\n".encode())
    # }

    # while True:
    #     try:
    #         x = input("输入动作(q退出/4夹紧/0松开/其他执行动作): ").strip()
            
    #         # 处理已知指令
    #         if x in command_actions:
    #             command_actions[x]()
    #             time.sleep(0.5)
    #             if x == "q":
    #                 break
    #             continue
                
    #         # 处理特殊动作指令
    #         if x in ["YIE", "3", "2", "智能抓取", "1"]:
    #             reset = False
    #             AC.Read_Track("json", 'config/motion_config.json', reset, command=x)
    #             AC.Run_Arm()
    #         else:
    #             print("未知指令")
                
    #     except Exception as e:
    #         print(f"发生错误: {e}")

    # # 清理资源
    # ser.close()
    # print("程序结束")