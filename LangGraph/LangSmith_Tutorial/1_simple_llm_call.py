from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOllama(model="llama3.2", temperature=0)
prompt = PromptTemplate.from_template("{question}")
parser = StrOutputParser()

# chain: prompt->llm->parser
chain = prompt | llm | parser

# result = chain.invoke({"question": "What is the capital of Peru?"})
result = chain.invoke({"question": "What is the capital of Nepal?"})
print(result)