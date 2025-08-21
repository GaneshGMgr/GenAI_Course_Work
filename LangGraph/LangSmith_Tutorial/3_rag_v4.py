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

# ---------- setup ----------
os.environ['LANGCHAIN_PROJECT'] = '3_rag_v4.py'
load_dotenv()

PDF_PATH = "data/islr.pdf"
INDEX_ROOT = Path("data/.indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ---------- traced helpers ----------
@traceable(name="load_pdf", tags=["io"], metadata={"type": "loader", "format": "pdf"})
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents", tags=["preprocess"], metadata={"type": "splitter"})
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore", tags=["index"], metadata={"type": "embedding_index"})
def build_vectorstore(splits, embed_model_name: str):
    embedding = SentenceTransformerEmbeddings(model_name=embed_model_name)
    return FAISS.from_documents(splits, embedding)

# ---------- fingerprinting ----------
# Create a unique ID for the PDF + settings so we can cache/reuse the index
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return { # Get file's SHA256 hash + size + last modified time
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = { # Combine: file fingerprint + chunking params + embedding model
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    # Then hash into one unique key for the index folder
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ---------- index load/build ----------
# Reuse saved index if it exists, otherwise build a fresh one
@traceable(name="load_index", tags=["index"], metadata={"mode": "load"})
def load_index_run(index_dir: Path, embed_model_name: str):
    embedding = SentenceTransformerEmbeddings(model_name=embed_model_name)
    return FAISS.load_local( # Load an existing FAISS index from local folder
        str(index_dir),
        embedding,
        allow_dangerous_deserialization=True,
    )

# 1. Load PDF → split → embed → build FAISS index
# 2. Save index + metadata locally for reuse
@traceable(name="build_index", tags=["index"], metadata={"mode": "build"})
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)  
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ---------- pipeline ----------
llm = ChatOllama(model="llama3.2", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"], metadata={"stage": "retriever_setup"})
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150, embed_model_name="all-MiniLM-L6-v2", force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="pdf_rag_full_run", tags=["qa"], metadata={"pipeline": "rag"})
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={"run_name": "pdf_rag_query", "tags": ["qa"], "metadata": {"k": 4}}
    )

# ---------- CLI ----------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    question = input("\nQuestion: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, question)
    print("\nAnswer: ", ans)

# Who is the author?
# What is the full form of GLM?