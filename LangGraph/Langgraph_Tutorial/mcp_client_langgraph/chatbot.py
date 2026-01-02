from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from model import llm_ollama, llm_chatOpenai

# Define a calculator tool using @tool
@tool
def calculator(first_num: int, second_num: int, operation: str) -> dict:
    """Perform basic arithmetic operations"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        elif operation == "mod":
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# Bind the tool to the LLM
tools = [calculator]
llm_with_tools = llm_ollama.bind_tools(tools)
llm_with_tools_chatOpenai = llm_chatOpenai.bind_tools(tools)

# Define LangGraph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools_chatOpenai.invoke(messages)
    return {"messages": [response]}

# Tool node
tool_node = ToolNode(tools)

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

# Run the graph
result = chatbot.invoke({
    "messages": [
        HumanMessage(
            content="Find the modulus of 132354 and 23 and give answer like a cricket commentator."
        )
    ]
})

# Print final response
print("=== Cricket Commentator Response ===")
print(result["messages"][-1].content)
