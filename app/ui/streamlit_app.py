import streamlit as st
from app.services.chatbot_service import FreeInsuranceChatbot
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="보험 챗봇", page_icon="🤖")

# Initialize chatbot with vector DB
chatbot = FreeInsuranceChatbot()

# Main page
st.title("보험 챗봇")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("질문을 입력하세요"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = chatbot.query(prompt)
            st.markdown(response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()
