import streamlit as st
from app.services.chatbot_service import SupabaseChatbot
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="보험 챗봇", page_icon="🤖")

# Initialize chatbot with Supabase
try:
    chatbot = SupabaseChatbot()
except ValueError as e:
    st.error("Supabase 연결에 실패했습니다. 환경 변수를 확인해주세요.")
    st.stop()

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
        with st.spinner("보험 상품 검색 중..."):
            response = chatbot.query(prompt)
            st.markdown(response, unsafe_allow_html=True)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()
