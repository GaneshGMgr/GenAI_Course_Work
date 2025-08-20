from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = '2_sequential_chain.py'

# First prompt: generate detailed report
prompt1 = PromptTemplate(
    # template='Generate a short report on {topic}',
    template = 'Write a short summary report (under 100 words) on {topic}.',
    input_variables=['topic']
)

# Second prompt: summarize into 2 points
prompt2 = PromptTemplate(
    template='Generate a 2 pointer summary from the following text:\n{text}',
    input_variables=['text']
)

# Two different models
llm1 = ChatOllama(model="llama3.2", temperature=0.7)   # report generation
llm2 = ChatOllama(model="llama3", temperature=0.5)       # summarization

# Output parser
parser = StrOutputParser()

# Chain: topic -> report (llm1) -> summary (llm2)
chain = prompt1 | llm1 | prompt2 | llm2 | parser

# Metadata for LangSmith / tracing
config = {
    'run_name': '2_sequential_chain',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {
        'llm1_model': 'llama3.2',
        'llm1_temp': 0.7,
        'llm2_model': 'llama3',
        'llm2_temp': 0.5,
        'parser': 'StrOutputParser'
    }
}

# Run chain with config
# result = chain.invoke({'topic': 'Unemployment in Nepal'}, config=config)
result = chain.invoke({'topic': 'Unemployment in World.'}, config=config)
print(result)
