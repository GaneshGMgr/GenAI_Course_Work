import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
from datetime import datetime
import time

# Custom CSS for styling
st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
        }
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 18px;
            margin-bottom: 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            font-size: 16px;
            line-height: 1.5;
        }
        
        /* User message specific styling */
        [data-testid="stChatMessage"][aria-label="user"] {
            background-color: #2563eb;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        /* Assistant message specific styling */
        [data-testid="stChatMessage"][aria-label="assistant"] {
            background-color: #f3f4f6;
            border-bottom-left-radius: 4px;
        }
        
        /* Chat input styling */
        .stChatInput {
            position: fixed;
            bottom: 20px;
            width: 70%;
            max-width: 800px;
            padding: 12px;
            background: white;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        }
        
        /* Timestamp styling */
        .timestamp {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 4px;
            display: block;
            text-align: right;
        }
        
        /* Title styling */
        .title {
            color: #2563eb;
            margin-bottom: 1.5rem;
        }
        
        /* Loading animation */
        .loading-dots {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px 0;
        }
        
        .loading-dots span {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #2563eb;
            margin: 0 3px;
            opacity: 0.4;
        }
        
        .loading-dots span:nth-child(1) {
            animation: pulse 1s infinite;
        }
        
        .loading-dots span:nth-child(2) {
            animation: pulse 1s infinite 0.2s;
        }
        
        .loading-dots span:nth-child(3) {
            animation: pulse 1s infinite 0.4s;
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
                opacity: 0.4;
            }
            50% {
                transform: scale(1.2);
                opacity: 1;
            }
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: inline-block;
            padding: 10px 15px;
            background-color: #f3f4f6;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            font-style: italic;
            color: #6b7280;
        }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="title">🤖 AI Chat Assistant</h1>', unsafe_allow_html=True)

# Configuration
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display conversation history with enhanced styling
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        # Add timestamp
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f'<span class="timestamp">{current_time}</span>', unsafe_allow_html=True)

# Chat input with better placeholder
user_input = st.chat_input('Type your message here...')

if user_input:
    # Add user message to history and display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f'<span class="timestamp">{current_time}</span>', unsafe_allow_html=True)

    # Show loading indicator
    with st.chat_message('assistant'):
        loading_placeholder = st.empty()
        
        # Option 1: Animated dots
        loading_html = """
        <div class="typing-indicator">
            AI is thinking
            <span class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </span>
        </div>
        """
        loading_placeholder.markdown(loading_html, unsafe_allow_html=True)

    # Get AI response
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]}, 
        config=CONFIG
    )
    
    ai_message = response['messages'][-1].content
    
    # Remove loading indicator and show actual response
    loading_placeholder.empty()
    
    # Add assistant response to history and display
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.markdown(ai_message)
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f'<span class="timestamp">{current_time}</span>', unsafe_allow_html=True)