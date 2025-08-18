import streamlit as st
from dotenv import load_dotenv
from app.services.chatbot_service import ChromaRAGChatbot

load_dotenv()

st.set_page_config(page_title="보험 RAG 챗봇", page_icon="💡")
st.title("🧠 보험 챗봇 (RAG 기반)")

try:
    chatbot = ChromaRAGChatbot()
except ValueError as e:
    st.error(str(e))
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바에 설정 옵션 추가
with st.sidebar:
    st.header("⚙️ 설정")

    # 모델 선택
    model_option = st.selectbox(
        "모델 선택",
        ["gpt-5-nano", "gpt-5-mini"],
        index=0,
        help="gpt-5-mini는 더 정확하지만 느립니다."
    )

    # 대화 히스토리 길이 설정
    max_history = st.slider(
        "대화 히스토리 길이",
        min_value=0,
        max_value=10,
        value=5,
        help="이전 대화를 몇 개까지 기억할지 설정합니다. 0이면 대화 히스토리를 사용하지 않습니다."
    )

    st.divider()

    # 대화 초기화 버튼
    if st.button("💬 대화 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # 통계 정보
    if st.session_state.messages:
        st.subheader("📊 대화 통계")
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.metric("사용자 질문", user_messages)
        st.metric("봇 답변", bot_messages)

# 이전 대화 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 입력
if prompt := st.chat_input("보험 관련 질문을 입력하세요…"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("문서 기반 답변 생성 중..."):
            # 대화 히스토리 준비 (max_history 개수만큼만)
            if max_history > 0:
                # 현재 질문을 제외한 이전 대화들
                recent_messages = st.session_state.messages[:-1]  # 현재 질문 제외
                # 최근 대화만 선택 (user-assistant 쌍으로)
                if len(recent_messages) > max_history * 2:
                    recent_messages = recent_messages[-(max_history * 2):]

                # 대화 히스토리를 문자열로 구성
                chat_history = []
                for i in range(0, len(recent_messages), 2):
                    if i + 1 < len(recent_messages):
                        user_msg = recent_messages[i]["content"]
                        bot_msg = recent_messages[i + 1]["content"]
                        chat_history.append(f"사용자: {user_msg}")
                        chat_history.append(f"어시스턴트: {bot_msg}")

                history_context = "\n".join(chat_history) if chat_history else ""
            else:
                history_context = ""

            # 대화 히스토리를 포함한 질문 생성
            if history_context and max_history > 0:
                enhanced_prompt = f"""이전 대화 내용:
{history_context}

현재 질문: {prompt}

위의 이전 대화 맥락을 고려하여 현재 질문에 답변해주세요."""
            else:
                enhanced_prompt = prompt

            # 모델에 따른 응답 생성
            try:
                response = chatbot.ask_with_detailed_context(enhanced_prompt, model=model_option)

                # 응답이 비어있는 경우 처리
                if not response or response["answer"].strip() == "":
                    response = "죄송합니다. 적절한 답변을 생성하지 못했습니다. 다시 질문해 주세요."

                st.markdown(response['answer'])

            except Exception as e:
                error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                response = error_msg

    # 어시스턴트 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})

# 하단에 도움말 표시
with st.expander("💡 사용법 및 팁"):
    st.markdown("""
    **기본 사용법:**
    - 보험 관련 질문을 자연어로 입력하세요
    - 이전 대화를 참고하여 연속적인 대화가 가능합니다

    **모델 선택:**
    - **gpt-5-nano**: 빠른 응답, 일반적인 질문에 적합

    **대화 히스토리:**
    - 사이드바에서 기억할 대화 개수를 조정할 수 있습니다
    - 0으로 설정하면 각 질문을 독립적으로 처리합니다

    **예시 질문:**
    - "어린이 실손보험의 보장 범위는?"
    - "방금 말한 보험의 보험료는 얼마인가요?" (연속 질문)
    - "다른 상품과 비교해서 어떤 차이점이 있나요?"
    """)

# 페이지 하단에 정보 표시
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🤖 현재 모델: {model_option}")
with col2:
    st.caption(f"💭 대화 히스토리: {max_history}개")
with col3:
    if st.session_state.messages:
        st.caption(f"📝 총 메시지: {len(st.session_state.messages)}개")