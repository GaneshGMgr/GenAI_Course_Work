import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = '3_rag_v1.py'


PDF_PATH = "data/islr.pdf"

# Load and split PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# Embeddings and vector store

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(splits, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Parallel runnable to combine question and context
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# Full chain
chain = parallel | prompt | llm | StrOutputParser()

# Metadata for LangSmith / tracing
config = {
    'run_name': '3_rag_v1',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {
        'llm_model': 'llama3.2',
        'llm_temp': 0,
        'parser': 'StrOutputParser'
    }
}

# Interactive loop
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    question = input("\nQuestion: ").strip()
    if not question:
        break
    answer = chain.invoke(question,  config=config)
    print("\nAnswer: ", answer)


### Testing Questions
# Who is the author of this book?