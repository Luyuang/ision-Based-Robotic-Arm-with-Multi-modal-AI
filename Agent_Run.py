import cv2
import re

from langchain_core.language_models.llms import LLM
from langchain_core.tools import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from util.Flask_Connect import *
from util.Qwen_Connect import *
from yolo.Yolo_Detection import Yolo
from Control_Arm import ArmControl
from Location_Capture import Rectangle_Pose, CameraParameter
from config.prompt_template import System_Arm_Use_Tool
from langchain_core.tools import tool


########################################################################################################################################################
#                                                                  Agent决策需要的工具                                                                  #
########################################################################################################################################################

@tool
def Execute_Action(motions):
    """这是接收动作,执行相应的动作命令"""
    # motion = [motion, **params]
    motion = motions[0]
    object = motions[1]["objects"]
    box_list = motions[1]["box_list"]
    # 预设动作
    if(motion in ["点头", "摇头", "跳舞", "立正", "右", "左"]):
        object.AC.Read_Track("json", 'config/motion_config.json', command=motion)
        object.AC.Run_Arm()
        
    # 进入抓取模式
    elif(motion in ["抓取模式",  "抓取"]):
        object.open_yolo = True
        if (motion in ["抓取模式"]):
            object.label = "None"
            object.capture = True
            object.AC.Read_Track("json", 'config/motion_config.json', False, command=motion)
            object.AC.Run_Arm()
            
        elif (motion in ["抓取"]):
            if object.label != "None" and box_list != [] and object.capture:
                pose = object.AC.Get_Pose()
                for i in box_list:
                    newpose = Rectangle_Pose(i, object.CP, pose[6:])
                    object.AC.Read_Track("list", newpose, True)
                    object.AC.Run_Arm(start_claw=True)
                box_list.clear()
            else:
                print(f"没有检测到{object.label},或者没有进入抓取模式")
    # 复位
    elif(motion in ["复位"]):
        object.AC.Update_Pose_Data()
        object.AC.Read_Track("list", object.AC.pose_config["zero_pose"], False)
        object.AC.Run_Arm()
        
    else:
        return "不在预设动作中"
    return motion
    
@tool
def Find_Object(object):
    """这是接收物体名字,返回标签对应的物体"""
    label = object[0]
    object = object[1]["objects"]
    #读取json，获取字典
    with open("config/contrast.json", 'r', encoding='utf-8') as f:
        path_dict = json.load(f)
    try:
        object.label = path_dict[label]
        object.open_yolo = True
    except:
        print("不认识这个类别")
    return object

########################################################################################################################################################
#                                                                 一些工具                                                                              #
########################################################################################################################################################


# 运行工具
def Invoke_Tool(tool_call_request, config = None, tools = None, **params):
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    arguments = tool_call_request["arguments"]
    arguments[list(arguments.keys())[0]] = [list(arguments.values())[0], params] # 在原参数的基础上添加其他参数
    
    return requested_tool.invoke(arguments, config=config)

# 格式化提示
def Format_Prompt(message : str) -> list:
    # 使用正则表达式匹配 "标记: 内容" 的结构
    pattern = r"(\bSystem\b|\bHuman\b|\bAI\b):\s*(.+?)(?=(\bSystem\b|\bHuman\b|\bAI\b|$))"
    matches = re.findall(pattern, message, re.DOTALL)
    message_list = []
    # 将提取的内容放入字典中
    for role, content, _ in matches:
        message_dict = {}
        if role == "System":
            message_dict["role"] = "user"
            message_dict["content"] = content.strip()
        elif role == "Human":
            message_dict["role"] = "user"
            message_dict["content"] = content.strip()
        elif role == "AI":
            message_dict["role"] = "assistant"
            message_dict["content"] = content.strip()
        message_list.append(message_dict)    
    return message_list

########################################################################################################################################################
#                                                                                                                                                      #
########################################################################################################################################################


# 自定义LLM类
class ChatModel(LLM):
    model_invoke_pattern : str
    
    def __init__(self, patterns: str,  **kwargs):
        super().__init__(model_invoke_pattern=patterns, **kwargs)  # 调用父类构造函数
        self.model_invoke_pattern = patterns  # 初始化实例属性
        
        
    def _call(self, prompt, stop = None, run_manager = None, **kwargs):
        message = Format_Prompt(prompt)
        if self.model_invoke_pattern == "Local":
            result =  Multi_Model_Message_to_Result(message)
        elif self.model_invoke_pattern == "Qwen":
            result =  Qwen_Chagt_to_Result(message)
        return result
    
    @property
    def _llm_type(self):
        return "ChatModel"


class ArmAgent():
    """这是机器人手臂的代理类"""
    def __init__(self, ArmControl, CameraParameter, tool, pattern):
        self.CM = ChatModel(patterns=pattern)
        
        self.YO = Yolo()
        self.YO.Initialize_Parames()
        self.YO.Initialize_Models()
        
        self.AC = ArmControl
        self.CP = CameraParameter
        self.AF = AudioFunc()
        
        self.open_yolo = False # 是否yolo检测
        self.label = "None"    # 需要检测标签
        self.box_list = []     # 检测到的所有框
        self.capture = False   # 是否进入抓取模式
        self.pattern = "Result" # 聊天的模式
        self.mindb = 3000 #音频录制的模式。4000为声音最小时暂停，True或者False为持续录音
        self.tools = tool
        self.rendered_tools = render_text_description(self.tools)

    # Agent调用
    def Invoke(self, question):
        prompt = ChatPromptTemplate.from_messages([("system", System_Arm_Use_Tool(self.rendered_tools)), ("human", "{input}")])
        use_tool = (prompt | self.CM | JsonOutputParser()).invoke({"input": question})
        # 调用工具
        result = [RunnablePassthrough.assign(output=lambda req: Invoke_Tool(req, tools=self.tools, objects=self, box_list=self.box_list)).invoke(i) for i in use_tool]
        # print("问题" + question)
        # print("工具" + str(use_tool))
        return use_tool

    # 开始聊天
    def Chat_Pattern(self, frame):
        # 只有本地部署的模型才可以使用
        if(self.pattern == "Vision"):
            self.AF.Transcribe_Audio("Temp/transcribe_audio.wav", self.mindb)
            cv2.imwrite('Temp/current_frame.jpg', frame)
            Multi_Model_Vision_to_Chat("Temp/transcribe_audio.wav", 'Temp/current_frame.jpg')
        # 只有本地部署的模型才可以使用
        elif(self.pattern == "Chat"):
            self.AF.Transcribe_Audio("Temp/transcribe_audio.wav", self.mindb)
            Multi_Model_Chat_to_Chat("Temp/transcribe_audio.wav")
                    
        # 可使用Qwen和local的API即可使用     
        elif(self.pattern == "Result"):
            if self.CM.model_invoke_pattern == "Local":
                self.AF.Transcribe_Audio("Temp/transcribe_audio.wav", self.mindb)
                question = Multi_Model_Audio_to_Text("Temp/transcribe_audio.wav").strip()
                self.Invoke(question)
            elif self.CM.model_invoke_pattern == "Qwen":
                self.AF.Transcribe_Audio("Temp/transcribe_audio.wav", self.mindb)
                question = Qwen_Audio_to_Text("Temp/transcribe_audio.wav").strip()
                self.Invoke(question)
            
    

    # 打开摄像头测试
    def Main(self):
        # 视频捕获设置
        cap = cv2.VideoCapture(0)  # 0是默认摄像头
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        while True:
            ret, frame = cap.read()
            frame_L = frame[:, 0:640]
            frame_R = frame[:, 640:1280]
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            if self.open_yolo:
                cv2.imwrite('Temp/current_frame.jpg', frame_R)
                self.YO.Set_Label(self.label)
                self.YO.Set_Source('Temp/current_frame.jpg')
                box_list_xyxy = self.YO.Perform_Inference(**vars(self.YO.opt))
                if box_list_xyxy != []:
                    self.box_list.clear()
                    for detection in box_list_xyxy:
                        # 提取边界框坐标、置信度和类别
                        x1, y1, x2, y2, conf, cls = detection
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = int(cls)  # 类别索引
                        label = self.YO.model.names[cls]  # 获取类别名称（例如 "apple"）

                        # 检查边界框是否在限制区域内
                        x, y, w, h = self.CP.limit_rectangle
                        rx, ry, rw, rh = x1, y1, x2 - x1, y2 - y1
                        if x1 > x and x1 < x + w and y1 > y and y1 < y + h and rx + rw <= x + w and ry + rh <= y + h:
                            # 绘制边界框
                            cv2.rectangle(frame_R, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # 绘制类别名称（在边界框上方）
                            label_text = f"{label} {conf:.2f}"  # 显示类别和置信度（例如 "apple 0.95"）
                            text_pos_y = max(y1 - 10, 10)  # 确保标签不超出图像顶部
                            cv2.putText(frame_R, label_text, (x1, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            self.box_list.append([rx, ry, rw, rh])
            
            # 相机矫正跟画限制框        
            u, v = frame_R.shape[:2]
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.CP.mtx, self.CP.dist, (u, v), 0, (u, v))
            dst_r = cv2.undistort(frame_R, self.CP.mtx, self.CP.dist, None, newcameramtx)
            cv2.rectangle(dst_r, (self.CP.limit_rectangle[0], self.CP.limit_rectangle[1]), (self.CP.limit_rectangle[0] + self.CP.limit_rectangle[2], self.CP.limit_rectangle[1] + self.CP.limit_rectangle[3]), (0, 255, 0), 2)
            
            cv2.imshow('Video Stream', dst_r)
            
            # 检测按键，开始录音
            if cv2.waitKey(1) & 0xFF == ord('r'):
                self.Chat_Pattern(dst_r)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 回到初始位置
                self.AC.Read_Track("list", self.AC.pose_config["zero_pose"], False)
                self.AC.Run_Arm()
                self.AC.claw_state.value = -1.0  # 关闭爪子线程
                break
                
if __name__ == '__main__':
    
    # 调用模型的方式
    AC = ArmControl()
    AC.print_targets = False
    AC.Set_Arm("COM10", claw_thread=False)
    
    CP = CameraParameter()
    # 初始化Agent机械臂
    tools = [Execute_Action, Find_Object]
    AA = ArmAgent(AC, CP, tools, AC.argument["model_invoke_pattern"])
    AA.Main()
    