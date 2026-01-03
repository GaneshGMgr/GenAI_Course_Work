from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="qwen3:latest", temperature=0.8)

# MCP client for local FastMCP server
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


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses for the month of Aug from 1 Aug to 30 Aug")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())