# model.py
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load environment variables
load_dotenv()
REQUIRED_ENV_VARS = [
    "UV_PATH",
    "MATH_SERVER_PATH",
    "MODEL_NAME",
    "OLLAMA_BASE_URL",
    "OPENAI_BASE_URL",
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

UV_PATH = os.environ["UV_PATH"]
PYTHON_PATH = os.environ["PYTHON_PATH"]
MATH_SERVER_PATH = os.environ["MATH_SERVER_PATH"]
EXPENSE_TRACKER_LOCAL_SERVER_PATH = os.environ["EXPENSE_TRACKER_LOCAL_SERVER_PATH"]
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "RJ7HGA0688TFYNE7")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:latest")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:latest")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# Initialize Ollama
llm_ollama = ChatOllama(
    model=MODEL_NAME,
    # model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.8,
)

# Initialize ChatOpenAI
llm_chatOpenai = ChatOpenAI(
    model=MODEL_NAME,
    base_url=OPENAI_BASE_URL,
    # api_key=OPENAI_API_KEY,
    temperature=0.8,
)

# Initialize OpenAI
llm_openai = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

# Optional: function to get both models
def get_models():
    return llm_ollama, llm_openai, llm_chatOpenai
