import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Title and header
st.markdown('<h1 class="title">🤖 AI Chat Assistant</h1>', unsafe_allow_html=True)

# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message('assistant'):

        # --- Stream the AI's response live ---
        ai_message = st.write_stream( # st.write_stream: write generator or streams to the app with typewriter effect.
            # Generator expression: gets the content of each chunk as it comes in
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},   # Pass the user's message to the bot},
                config={'configurable': {'thread_id': 'thread-1'}}, # set a thread ID for stateful conversations
                stream_mode='messages'                              # Tells the bot to stream messages (not tokens)
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})