from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
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
import sys
import os

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import llm_ollama, UV_PATH

load_dotenv()

# ==========================
# Monkey-patch aiosqlite.Connection
# ==========================
# This is required because AsyncSqliteSaver expects an 'is_alive()' method
def dummy_is_alive(self):
    return True

aiosqlite.Connection.is_alive = dummy_is_alive

# ==========================
# Async loop for backend tasks
# ==========================
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ==========================
# Tools
# ==========================
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price using Alpha Vantage API."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    response = requests.get(url)
    return response.json()


# ==========================
# MCP client configuration
# ==========================
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
    """Synchronously load MCP tools from servers."""
    try:
        return run_async(client.get_tools())
    except Exception as exc:
        print("Failed to load MCP tools:", exc)
        return []


mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]

llm_with_tools = llm_ollama.bind_tools(tools) if tools else llm_ollama


# ==========================
# LangGraph state
# ==========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    # response = run_async(llm_with_tools.ainvoke(messages))
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None


# ==========================
# Async checkpointer
# ==========================
async def _init_checkpointer():
    conn = await aiosqlite.connect("chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())


# ==========================
# Graph
# ==========================
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


# ==========================
# Helper functions
# ==========================
async def _alist_threads():
    """List all thread IDs from checkpointer."""
    thread_ids = set()
    async for checkpoint in checkpointer.alist(None):
        thread_ids.add(
            checkpoint.config["configurable"]["thread_id"]
        )
    return list(thread_ids)


def retrieve_all_threads():
    """Synchronously retrieve all threads."""
    return run_async(_alist_threads())
