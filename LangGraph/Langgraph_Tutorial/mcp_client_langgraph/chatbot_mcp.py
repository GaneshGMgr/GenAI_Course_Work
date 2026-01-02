from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from model import llm_ollama, UV_PATH

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
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

# MCP Client setup for calculator tools
SERVERS: dict[str, Connection] = {
    "math": {
        "transport": "stdio",
        "command": UV_PATH,
        "args": [
            "--directory",
            "C:\\Users\\ganes_3ck5\\DataScience\\Gen_AI\\Course_GenAI\\Gen_AI_In-Depth\\MCP_Model_Context_Protocol\\chatbot_with_mcp_server",
            "run",
            "fastmcp",
            "run",
            "math.py",
        ],
    },
    "expense_tracker": {
        "transport": "stdio",
        "command": UV_PATH,
        "args": [
            "--directory",
            "C:\\Users\\ganes_3ck5\\DataScience\\Gen_AI\\Course_GenAI\\Gen_AI_In-Depth\\MCP_Model_Context_Protocol\\expense_tracker_local_mcp_server",
            "run",
            "--isolated",
            "fastmcp",
            "run",
            "main.py",
        ],
    },
}

client = MultiServerMCPClient(SERVERS)

# Define LangGraph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    tools = await client.get_tools()
    llm_with_tools = llm_ollama.bind_tools(tools)
    # Chat node
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
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
    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({
        "messages": [
            SYSTEM_PROMPT,
            HumanMessage(
                # content="First add 3456 and 7890, then find the modulus of the result with 97."
                # content="What is the remainder when 456789 is divided by 37?"
                # content="Add an expense of $50 for groceries and $20 for transport on March 1st."
                # content="Add $30 for groceries on December 3rd. Note: bought snacks and fruits."
                # content="Summarize all expenses for October 2024."
                content="Change the amount of expense ID 5 to $75.50 and add a note saying 'Bought snacks'"
            )
        ]
    })

    # Print final response
    print("=== Response ===")
    # print("Available Tools: ", await client.get_tools())
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())