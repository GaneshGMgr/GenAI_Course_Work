# # Regular function: returns a list all at once
# def get_numbers():
#     return [1, 2, 3]  # All values are returned at once

# print("Regular function outputs:", get_numbers())

# # Generator function: yields one value at a time
# def generate_numbers():
#     for i in range(1, 4):
#         yield i  # Pauses here and resumes on next() call

# gen = generate_numbers()  # gen is a generator object

# # Use next() to get values one-by-one
# print("Generator outputs:")
# print(next(gen))  # ➜ 1
# print(next(gen))  # ➜ 2
# # print(next(gen))  # ➜ 3

# # Decorator function: A decorator is a function that takes another function as input and returns a new function with extra behavior.
# def log_calls(func):
#     def wrapper(*args, **kwargs):
#         print(f"📞 Calling: {func.__name__} function")
#         result = func(*args, **kwargs) # Call the original greet() function
#         print(f"✅ Finished: {func.__name__} function")
#         return result
#     return wrapper

# @log_calls
# def greet(name):
#     print(f"Hello, {name}!")

# greet("Alice")



import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ================================== Utility Functions ==================================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []
    # Start with empty title
    st.session_state['thread_titles'][thread_id] = ""

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

# ================================== Session Initialization ==================================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

# Ensure current thread exists
add_thread(st.session_state['thread_id'])

# Ensure current thread exists
if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])

# Initialize title if not exists (empty string)
if st.session_state['thread_id'] not in st.session_state['thread_titles']:
    st.session_state['thread_titles'][st.session_state['thread_id']] = ""

# ================================== Sidebar UI ==================================
st.sidebar.title('Chat Sessions')

if st.sidebar.button('➕ New Chat'):
    reset_chat()

st.sidebar.header('History')
for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['thread_titles'].get(thread_id, "")
    if title:
        if thread_id == st.session_state['thread_id']:
            # Use emoji + disabled button to current active chat
            col1, col2 = st.sidebar.columns([0.1, 0.8])
            col1.markdown("▶️")
            col2.button(title, disabled=True, key=f"active_{thread_id}")
        else:
            if st.sidebar.button(title, key=f"chat_{thread_id}"):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                st.session_state['message_history'] = [
                    {'role': 'user' if isinstance(msg, HumanMessage) else 'assistant', 
                     'content': msg.content}
                    for msg in messages
                ]
                st.rerun()

# ================================== Main Chat UI ==================================
# Display message history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Chat input
user_input = st.chat_input('Type your message...')

if user_input:
    # Update title if this is the first message
    if not st.session_state['message_history']:
        st.session_state['thread_titles'][st.session_state['thread_id']] = (
            user_input[:16] + "..." if len(user_input) > 16 else user_input
        )
    
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.write(user_input)

    # Get AI response
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    with st.chat_message('assistant'):
        ai_response = st.write_stream(
            message_chunk.content for message_chunk, _ in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )
    
    # Add AI response to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})