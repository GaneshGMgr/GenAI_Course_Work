from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os

# Load environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

# Initialize local LLM
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3:latest",
    temperature=0.0,
)

# Tool: fetch stock price with FMP first, fallback to Alpha Vantage
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol.
    Tries Financial Modeling Prep first; falls back to Alpha Vantage if FMP fails.
    
    :param symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    :return: Dictionary with source, symbol, price, and volume
    """
    try:
        # FMP API
        data = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey=demo",
            timeout=5
        ).json()
        if data and "price" in data[0]:
            return {
                "source": "FMP",
                "symbol": data[0]["symbol"],
                "price": data[0]["price"],
                "volume": data[0]["volume"]
            }
    except Exception:
        pass

    # Fallback Alpha Vantage
    try:
        quote = requests.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM",
            timeout=5
        ).json().get("Global Quote", {})
        if quote:
            return {
                "source": "Alpha Vantage",
                "symbol": quote.get("01. symbol"),
                "price": float(quote.get("05. price", 0)),
                "volume": int(quote.get("06. volume", 0))
            }
    except Exception:
        pass

    return {"error": "Unable to fetch stock price from both APIs."}

# Tool: simulate purchasing stock
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a stock order.

    :param symbol: Stock ticker symbol (e.g., 'AAPL')
    :param quantity: Number of shares to purchase
    :return: Dictionary confirming the purchase with details
    """
    return {
        "status": "success",
        "message": f"Purchased {quantity} shares of {symbol}",
        "symbol": symbol,
        "quantity": quantity
    }

# Bind tools to LLM
tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# Define Chat State
class ChatState(TypedDict):
    user_input: Annotated[str, "The user's input message"]
    messages: Annotated[list[BaseMessage], add_messages]

# LLM Node
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"] + [HumanMessage(content=state["user_input"])]
    response = llm_with_tools.invoke(input=messages)
    return {"user_input": state["user_input"], "messages": response}

# Tool Node
tool_node = ToolNode(tools)

# Memory saver
memory = MemorySaver()

# Build Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")  # corrected edge

# Compile chatbot
chatbot = graph.compile(checkpointer=memory)

# CLI Loop
if __name__ == "__main__":
    print("Welcome to the Stock Trading Chatbot! Type 'exit' to quit.")
    thread_id = "user_thread_1"
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state
        state = {"user_input": user_input, "response": []}

        # Run graph
        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # Get latest assistant message
        messages = result["messages"]
        if messages:
            last_msg = messages[-1]
            print(f"Bot: {last_msg.content}\n")
        else:
            print("Bot: [No response]\n")
