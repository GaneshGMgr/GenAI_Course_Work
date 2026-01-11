# model.py

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os

# -----------------------------
# Setup
# -----------------------------
# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load environment variables
load_dotenv()

REQUIRED_ENV_VARS = [
    "MODEL_NAME",
    "OLLAMA_BASE_URL",
    "OPENAI_BASE_URL",
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# API Keys & Model Names
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "RJ7HGA0688TFYNE7")
REASONING_MODEL = os.getenv("MODEL_NAME", "qwen3:latest")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:latest")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gemma3:4b")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# -----------------------------
# Initialize LLMs
# -----------------------------

# Ollama
llm_ollama = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.8,
)

# ChatOpenAI
llm_chat_openai = ChatOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=REASONING_MODEL,
    temperature=0.0,
)

# OpenAI client
llm_openai = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

# -----------------------------
# Helper function
# -----------------------------
def get_models():
    """
    Returns all initialized LLMs:
    - llm_ollama
    - llm_openai
    - llm_chat_openai
    """
    return llm_ollama, llm_openai, llm_chat_openai
