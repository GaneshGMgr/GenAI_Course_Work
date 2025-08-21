# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
import json
import hashlib
from pathlib import Path
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

os.environ['LANGCHAIN_PROJECT'] = '3_rag_v4.py'
load_dotenv()

PDF_PATH = "data/islr.pdf"
INDEX_ROOT = Path(".data/indices")
INDEX_ROOT.mkdir(exist_ok=True)
# all-MiniLM-L6-v2
# ------------ helpers (traces) ------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size = 1000, chunk_overlap = 150):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name = "build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    embedding = SentenceTransformerEmbeddings(model = embed_model_name)
    return FAISS.from_documents(splits, embedding)

# ----------------- cache key / fingerprint ------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str)-> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys = True).encode("utf-8")).hexdigest()

# -------------- explicitly traced load/build runs --------------
@traceable(name="load_index", tags=["inded"])
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = SentenceTransformerEmbeddings(model=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
