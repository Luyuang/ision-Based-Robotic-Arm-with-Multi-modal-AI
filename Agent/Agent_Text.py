import re
import sys
sys.path.append("f:\\Learning\\Multi_Model_Arm")

from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel 


from util.Flask_Connect import *
from config.prompt_template import System_Get_Use_Tool, Ai_Simple_Reply, System_Arm_Use_Tool
from config.tools import *


# 自定义LLM类
class GlmModel(LLM, BaseChatModel):
    def __init__(self):
        super().__init__()
    
    
    def _call(self, prompt, stop = None, run_manager = None, **kwargs):
        message = Format_Prompt(prompt)
        result =  Multi_Model_Message_to_Result(message)
        return result
    
    @property
    def _llm_type(self):
        return "ChatModel"
    
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

# 运行工具
def Invoke_Tool(
    tool_call_request, config = None, tools = None
):
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)


if __name__ == "__main__":
    model = GlmModel()
    too = [Add, Multiply]
    question = "计算 2+3 和 3 * 4"
    
    rendered_tools = render_text_description(too)
    prompt = ChatPromptTemplate.from_messages(
        [("system", System_Get_Use_Tool(rendered_tools)), ("human", "{input}")]
    )
    use_tool = (prompt | model | JsonOutputParser()).invoke({"input": question})
    print(use_tool)
    result = [RunnablePassthrough.assign(output=lambda req: Invoke_Tool(req, tools=too)).invoke(i) for i in use_tool]
    results = [i["output"] for i in result]
    new_prompt = ChatPromptTemplate.from_messages(
        [("ai", Ai_Simple_Reply(results)), ("human", "{input}")]
    )
    new_chain = new_prompt | model 
    d = new_chain.invoke({"input": question})
    print(d)
    
    

    
    
    