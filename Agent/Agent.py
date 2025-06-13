# Import relevant functionality
from Agent_Text import GlmModel 
from langchain_core.tools import tool

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


# Create the agent
model = GlmModel()
tools = [add, multiply]
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

x = agent_executor.invoke({"input": "计算一下3 * 4 和 5 + 6"})
print(x["output"])


