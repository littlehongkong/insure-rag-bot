import os
import streamlit as st
from dotenv import load_dotenv
from app.services.chatbot_service import SupabaseRAGChatbot

load_dotenv()

st.set_page_config(page_title="보험 RAG 챗봇", page_icon="💡")
st.title("🧠 보험 챗봇 (RAG 기반)")

try:
    chatbot = SupabaseRAGChatbot()
except ValueError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 입력
if prompt := st.chat_input("보험 관련 질문을 입력하세요…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("문서 기반 답변 생성 중..."):
            response = chatbot.query(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 대화 초기화 버튼
with st.sidebar:
    if st.button("💬 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
