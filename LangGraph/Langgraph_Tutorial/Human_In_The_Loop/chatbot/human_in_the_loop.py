from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os

# -------------------
# Environment Setup
# -------------------
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
load_dotenv()

# -------------------
# Initialize Local LLM
# -------------------
llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.0)),
)

# -------------------
# Tools
# -------------------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol.
    Tries Financial Modeling Prep first, then Alpha Vantage if FMP fails.

    :param symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    :return: Dictionary with source, symbol, price, and volume
    """
    try:
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


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a stock.
    HUMAN-IN-THE-LOOP: asks user for confirmation.
    """
    # HITL interrupt
    decision = interrupt(f"Do you want to purchase {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"The purchase order for {quantity} shares of {symbol} has been successfully placed! 🎉",
            "symbol": symbol,
            "quantity": quantity
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was cancelled by the user.",
            "symbol": symbol,
            "quantity": quantity
        }

# Bind tools to LLM
tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# Define Chat State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# LLM Node
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# Memory Saver
# -------------------
memory = MemorySaver()

# -------------------
# Build Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)

# -------------------
# CLI Loop with HITL
# -------------------
if __name__ == "__main__":
    print("Welcome to the Stock Trading Chatbot! Type 'exit' to quit.")
    thread_id = "user_thread_1"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Initialize state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Check for interrupts
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()
        
            # Resume with Command(resume=decision)
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        # Print final bot response
        messages = result["messages"]
        if messages:
            last_msg = messages[-1]
            print(f"Bot: {last_msg.content}\n")
        else:
            print("Bot: [No response]\n")
