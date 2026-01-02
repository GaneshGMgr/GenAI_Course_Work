from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from model import llm_ollama, llm_chatOpenai
import asyncio

# Prompt
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a precise assistant that solves mathematical problems step by step. "
        "You always use tools when calculations are required. "
        "You explain your reasoning clearly as if teaching a student, "
        "and you return accurate final results."
    )
)

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

def build_graph():
    # Chat node
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools_chatOpenai.ainvoke(messages)
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
    
    return chatbot

async def main():
    chatbot = build_graph()

    # running the graph
    result = await chatbot.ainvoke({
        "messages": [
            SYSTEM_PROMPT,
            HumanMessage(
                content="First add 3456 and 7890, then find the modulus of the result with 97."
            )
        ]
    })

    # Print final response
    print("=== Response ===")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())