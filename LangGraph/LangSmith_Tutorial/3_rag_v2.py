# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = '3_rag_v2.py'

PDF_PATH = "data/islr.pdf"

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)  # need to initialize with path
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

# Umbrella traced setup
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vector_store = build_vectorstore(splits)
    return vector_store

# ---------- pipeline ----------
llm = ChatOllama(model="llama3.2", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build vectorstore and retriever
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # lowercase 'k'
)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})
chain = parallel | prompt | llm | StrOutputParser()

# ---------- interactive query ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    question = input("\nQuestion: ").strip()
    if not question:
        break

    # Metadata for LangSmith tracing
    config = {
        "run_name": "3_rag_v2",
        "tags": ["rag", "pdf", "ollama", "faiss"],
        "metadata": {
            "llm_model": "llama3.2",
            "llm_temp": 0
        }
    }

    answer = chain.invoke(question, config=config)
    print("\nAnswer: ", answer)
