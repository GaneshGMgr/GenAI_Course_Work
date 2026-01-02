from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os

load_dotenv()

UV_PATH = os.environ["UV_PATH"]
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

# ------------------- Async helpers -------------------
def dummy_is_alive(self):
    return True
aiosqlite.Connection.is_alive = dummy_is_alive

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# ------------------- LLM -------------------
llm_ollama = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.8,
)

# ------------------- Tools -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price using Alpha Vantage API"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": UV_PATH,
        "args": [
            "--directory",
            r"C:\Users\ganes_3ck5\DataScience\Gen_AI\Course_GenAI\Gen_AI_In-Depth\MCP_Model_Context_Protocol\chatbot_with_mcp_server",
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
            r"C:\Users\ganes_3ck5\DataScience\Gen_AI\Course_GenAI\Gen_AI_In-Depth\MCP_Model_Context_Protocol\expense_tracker_local_mcp_server",
            "run",
            "--isolated",
            "fastmcp",
            "run",
            "main.py",
        ],
    },
}

client = MultiServerMCPClient(SERVERS)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm_ollama.bind_tools(tools) if tools else llm_ollama

# ------------------- System Prompt -------------------
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a precise assistant that solves problems step by step. "
        "Always use available tools whenever a calculation or task is required, "
        "even if you can compute it yourself. "
        "Whenever a tool returns structured data (e.g., JSON), extract the relevant information "
        "and present it in a human-friendly way. "
        "Explain your reasoning clearly as if teaching a student, "
        "and always provide accurate final results."
    )
)
# ------------------- State -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------- Chat Node -------------------
async def chat_node(state: ChatState):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke([SYSTEM_PROMPT] + messages)

    # Create a dictionary to capture tool info
    # tool_info = None
    # if hasattr(response, "tool_used") and response.tool_used:
    #     tool_info = {
    #         "tool_name": response.tool_used,
    #         "tool_input": response.tool_input,
    #         "tool_output": response.tool_output,
    #     }
    #     print(f"Tool called: {response.tool_used}")
    #     print(f"Tool input: {response.tool_input}")
    #     print(f"Tool output: {response.tool_output}")

    # print("LLM response content:", getattr(response, "content", response))

    # Return both LLM response and tool info
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# ------------------- Checkpointer -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# ------------------- Graph -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ------------------- Helper -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

# ------------------- Available Tools -------------------
print("Available tools:")
for t in tools:
    print(f"- {t.name}")
