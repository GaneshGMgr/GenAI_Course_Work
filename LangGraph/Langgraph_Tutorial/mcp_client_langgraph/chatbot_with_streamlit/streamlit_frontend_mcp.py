import uuid
import streamlit as st

from langgraph_mcp_backend import chatbot, retrieve_all_threads
from langchain_core.messages import AIMessage, HumanMessage

# --------------------------
# Utilities
# --------------------------
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def add_thread(thread_id: str, title: str = None):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []

    # Avoid duplicates
    if any(thread["id"] == thread_id for thread in st.session_state["chat_threads"]):
        return

    display_title = title if title else "New Chat"
    st.session_state["chat_threads"].append({"id": thread_id, "title": display_title})

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id, title="New Chat")
    st.session_state["message_history"] = []

def load_conversation(thread_id: str):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])

# --------------------------
# Session initialization
# --------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Wrap string thread_ids into dicts
    st.session_state["chat_threads"] = [
        {"id": tid, "title": "New Chat"} for tid in retrieve_all_threads()
    ]

# Add current thread if not exists
add_thread(st.session_state["thread_id"])

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread in reversed(st.session_state["chat_threads"]):
    thread_id = thread["id"]
    title = thread["title"]
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        # Update title with first user message
        if messages:
            first_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), None)
            if first_msg:
                thread["title"] = first_msg[:50]

        # Load history
        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = history

# --------------------------
# Main UI
# --------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.text(user_input)

    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
