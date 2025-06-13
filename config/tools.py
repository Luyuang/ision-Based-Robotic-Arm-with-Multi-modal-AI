import json
import math
import threading

from Location_Capture import Rectangle_Pose
from langchain_core.tools import tool

@tool
def Add(a, b):
    """相加a和b,a、b的类型可以是int、float"""
    return a + b

@tool
def Multiply(a, b):
    """相乘a和b,a、b的类型可以是int、float"""
    return a * b

##################################################################################
#                               GUI的Agent调用的工具                              #
##################################################################################

@tool
def Execute_Action(motions):
    """这是接收动作,执行相应的动作命令"""
    # motion = [motion, **params]
    motion = motions[0]
    objects = motions[1]["objects"]
    box_list = motions[1]["box_list"]
    # 预设动作
    if(motion in ["点头", "摇头", "跳舞", "立正", "右", "左"]):
        objects.AC.Read_Track("json", 'config/motion_config.json', command=motion)
        thread = threading.Thread(target=objects.Action_Arm, args=(objects.AC.Run_Arm, None, None, False, "Agent模式", "agent"))
        thread.start()
        
    # 进入抓取模式
    elif(motion in ["抓取模式",  "抓取"]):
        objects.AA.open_yolo = True
        if (motion in ["抓取模式"]):
            objects.AA.label = "None"
            objects.AA.capture = True
            objects.AC.Read_Track("json", 'config/motion_config.json', False, command=motion)
            thread = threading.Thread(target=objects.Action_Arm, args=(objects.AC.Run_Arm, None, None, False, "Agent模式", "agent"))
            thread.start()
            
        elif (motion in ["抓取"]):
            if objects.label != "None" and box_list != [] and objects.AA.capture:
                pose = objects.AC.Get_Pose()
                for i in box_list:
                    newpose = Rectangle_Pose(i, objects.CP, pose[6:])
                    objects.AC.Read_Track("list", newpose, True)
                thread = threading.Thread(target=objects.Action_Arm, args=(objects.AC.Run_Arm, None, None, True, "Agent模式", "agent"))
                thread.start()
                box_list.clear()
            else:
                print(f"没有检测到{objects.AA.label},或者没有进入抓取模式")
    # 复位
    elif(motion in ["复位"]):
        objects.AC.Update_Pose_Data()
        objects.AC.Read_Track("list", objects.AC.pose_config["zero_pose"], False)
        thread = threading.Thread(target=objects.Action_Arm, args=(objects.AC.Run_Arm, None, None, False, "Agent模式", "agent"))
        thread.start()
        
    else:
        return None
    return motion
    
@tool
def Find_Object(object):
    """这是接收物体名字,返回标签对应的物体"""
    label = object[0]
    objects = object[1]["objects"]
    #读取json，获取字典
    with open("config/contrast.json", 'r', encoding='utf-8') as f:
        path_dict = json.load(f)
    try:
        objects.AA.label = path_dict[label]
        objects.AA.open_yolo = True
    except:
        print("不认识这个类别")
        return None
    return object