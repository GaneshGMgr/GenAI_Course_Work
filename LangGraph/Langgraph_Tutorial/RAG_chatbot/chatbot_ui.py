import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_rag import (
    load_pdf_documents,
    rag_chatbot,
    retrieve_all_threads,
    thread_document_metadata,
)

# -----------------------------
# Thread / Session helpers
# -----------------------------
def generate_thread_id() -> str:
    return uuid.uuid4().hex


def add_thread(thread_id: str):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id: str):
    state = rag_chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# -----------------------------
# Session initialization
# -----------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.title("LangGraph RAG Chatbot")

# New Chat button
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Current thread info
st.sidebar.markdown(f"**Current Thread:** `{thread_key[:8]}...`")

# Latest document in current thread
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"""
**Latest uploaded document**
- **Filename:** {latest_doc.get("filename", "unknown")}
- **Chunks:** {latest_doc.get("num_chunks", "unknown")}
- **Pages:** {latest_doc.get("num_pages", "unknown")}
"""
    )
else:
    st.sidebar.info("No documents uploaded yet.")

# -----------------------------
# PDF Upload with "Upload" button
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Select PDF", type=["pdf"])

# Store temporarily in session state
if uploaded_file:
    st.session_state["pending_upload"] = uploaded_file

# Only process when user clicks the button
if st.session_state.get("pending_upload"):
    if st.sidebar.button("Upload & Process PDF"):
        file_to_process = st.session_state.pop("pending_upload")
        if file_to_process.name in thread_docs:
            st.sidebar.warning(f"`{file_to_process.name}` already uploaded.")
        else:
            with st.spinner("Loading and ingesting document..."):
                summary = load_pdf_documents(
                    file_to_process.getvalue(),
                    thread_id=thread_key,
                    filename=file_to_process.name,
                )
                thread_docs[file_to_process.name] = summary
            st.sidebar.success("Document ingested successfully!")

# -----------------------------
# Past conversations preview
# -----------------------------
st.sidebar.subheader("Past Conversations")
threads = st.session_state.get("chat_threads", [])[::-1]

if not threads:
    st.sidebar.info("No past conversations yet. Start a new chat!")
else:
    for tid in threads:
        messages = load_conversation(tid)
        first_message_preview = ""
        if messages:
            first_msg = messages[0]
            if isinstance(first_msg, (HumanMessage, AIMessage)):
                first_message_preview = first_msg.content
            # truncate
            first_message_preview = first_message_preview[:25] + ("..." if len(first_message_preview) > 25 else "")

        # Document info for preview
        doc_meta = thread_document_metadata(tid)
        doc_preview = f" 📄 {doc_meta.get('filename')}" if doc_meta else ""

        label = first_message_preview or f"Thread {tid[:8]}..."
        label += doc_preview

        if st.sidebar.button(label, key=f"thread_{tid}"):
            st.session_state["selected_thread"] = tid
            st.rerun()

# -----------------------------
# Main chat UI
# -----------------------------
st.title("Multi-Utility RAG Chatbot (LangGraph)")

# Show conversation messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about your documents, calculations, or stock prices…")

# -----------------------------
# Chat execution
# -----------------------------
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_box = None
        response_placeholder = st.empty()

        def ai_stream():
            global status_box
            for message, _ in rag_chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message, ToolMessage):
                    tool_name = message.name or "tool"
                    if status_box is None:
                        status_box = st.status(f"Using `{tool_name}`…", expanded=True)
                    else:
                        status_box.update(label=f"Using `{tool_name}`…", state="running")

                if isinstance(message, AIMessage):
                    yield message.content

        ai_text = st.write_stream(ai_stream)

        if status_box is not None:
            status_box.update(label="Tool finished", state="complete", expanded=False)

        st.session_state["message_history"].append({"role": "assistant", "content": ai_text})

        # -------------------------
        # Display metadata / source
        # -------------------------
        doc_meta = thread_document_metadata(thread_key)
        st.markdown(ai_text)  # show the AI answer
    
        # Display sources / references if available
        if doc_meta:
            st.caption(
                f"Source / Reference: `{doc_meta.get('filename', 'unknown')}` | "
                f"Pages: {doc_meta.get('num_pages', '?')}, "
                f"Chunks: {doc_meta.get('num_chunks', '?')}"
            )
    
        st.divider()

# -----------------------------
# Load selected thread
# -----------------------------
if "selected_thread" in st.session_state:
    selected_thread = st.session_state["selected_thread"]
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    rebuilt = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        rebuilt.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = rebuilt
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
