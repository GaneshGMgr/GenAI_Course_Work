# LangGraph Installation & References

## Installation

```bash
uv init .
uv venv

# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=langgraph-tutorial --display-name "Python 3.13 LangGraph"

uv add streamlit --active
uv add langgraph --active
uv add langchain --active
uv add langchain-core
uv add langchain-community
uv add langchain-ollama
uv add faiss-cpu
uv add pypdf
uv add python-dotenv
uv add huggingface-hub
pip install -qU  langchain langchain-huggingface sentence_transformers
uv add duckduckgo-search
ollama pull qwen3

```

## Documentation & References

* **OLLAMA Tool Calling**: [https://docs.ollama.com/capabilities/tool-calling#python](https://docs.ollama.com/capabilities/tool-calling#python)
tool support ollama: https://ollama.com/blog/tool-support?utm_source=chatgpt.com
* **OpenAI Compatibility**: [https://docs.ollama.com/api/openai-compatibility](https://docs.ollama.com/api/openai-compatibility)
* **Ollama Documentation**: [https://docs.ollama.com/](https://docs.ollama.com/)
* **langchain-ollama**: [https://reference.langchain.com/python/integrations/langchain_ollama/?_gl=1*1c60x9x*_gcl_au*MjExODM5NzQ2NC4xNzY1OTk0MzU5*_ga*NDQwMjI2MjY5LjE3NjU5OTQzNjA.*_ga_47WX3HKKY2*czE3NjYwMDExMTUkbzIkZzEkdDE3NjYwMDE4MTQkajYwJGwwJGgw#langchain_ollama.ChatOllama](https://reference.langchain.com/python/integrations/langchain_ollama/?_gl=1*1c60x9x*_gcl_au*MjExODM5NzQ2NC4xNzY1OTk0MzU5*_ga*NDQwMjI2MjY5LjE3NjU5OTQzNjA.*_ga_47WX3HKKY2*czE3NjYwMDExMTUkbzIkZzEkdDE3NjYwMDE4MTQkajYwJGwwJGgw#langchain_ollama.ChatOllama)
