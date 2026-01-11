# Environment & Imports
import os
import tempfile
import sqlite3
import requests
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
# Document loading & processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & vector stores
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain tools
from langchain_core.tools import tool

# LangGraph core
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Messages
from langchain_core.messages import SystemMessage, BaseMessage

# Custom LLMs
from langchain_community.tools import DuckDuckGoSearchRun
from model import llm_chat_openai

load_dotenv()

_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {} 

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# PDF Loading & Retriever
def load_pdf_documents(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever from uploaded PDF and store it for a thread."""
    if not file_bytes:
        raise ValueError("No file bytes provided for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(texts, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "source": "pdf_upload",
            "filename": filename or os.path.basename(temp_pdf_path) or "uploaded_document.pdf",
            "num_pages": len(documents),
            "num_chunks": len(texts),
        }

        return {
            "filename": filename or os.path.basename(temp_pdf_path) or "uploaded_document.pdf",
            "num_pages": len(documents),
            "num_chunks": len(texts),
        }
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass


# Tools
@tool
def calculator(first_num: int, second_num: int, operation: str) -> dict:
    """Perform basic arithmetic operations: add, sub, mul, div, mod."""
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

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def retrieve_documents(query: str, thread_id: str) -> dict:
    """Retrieve relevant content and metadata from uploaded PDF documents in the given thread."""
    retriever = _THREAD_RETRIEVERS.get(str(thread_id))
    if retriever is None:
        return {
            'query': query,
            'combined_content': "No documents have been uploaded in this thread.",
            'metadata': []
        }

    result = retriever.invoke(query)
    combined_content = "\n\n".join(doc.page_content for doc in result)
    metadata = [f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})" for doc in result]

    return {
        'query': query,
        'combined_content': combined_content,
        'metadata': metadata,
        'source_file': _THREAD_METADATA.get(str(thread_id), {}).get("filename", "unknown"),
    }

search_tool = DuckDuckGoSearchRun(region="us-en")
# Bind tools to LLM
tools = [calculator, get_stock_price, retrieve_documents, search_tool]
llm_with_tools = llm_chat_openai.bind_tools(tools)


# ChatState and Node
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about uploaded PDFs, call "
            "the `retrieve_documents` tool and include the thread_id "
            f"`{thread_id}`. You can also use web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)


# Database & Checkpointer
conn = sqlite3.connect(database='chatbot_memory.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

rag_chatbot = graph.compile(checkpointer=checkpointer)


# Helpers
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_documents(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})