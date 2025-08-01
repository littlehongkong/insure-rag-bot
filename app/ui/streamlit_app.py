import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Page configuration
st.set_page_config(
    page_title="보험 챗봇",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chatbot
def init_chatbot():
    # Initialize embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Initialize Chroma DB
    persist_directory = "./chroma_db"
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Initialize Ollama
    llm = Ollama(
        model="llama2",
        temperature=0.1
    )

    # Custom prompt template
    prompt_template = """
    당신은 보험 약관을 분석하는 전문가입니다.
    다음 내용을 참고하여 질문에 정확하게 답해주세요:

    [보장 항목 발췌]
    {context}

    [사용자 질문]
    {question}

    답변을 한국어로 간결하고 정확하게 제공해주세요.
    모르는 내용은 모른다고 답변해주세요.
    
    답변:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# Initialize chatbot
if "qa_chain" not in st.session_state:
    with st.spinner("챗봇을 초기화하는 중..."):
        try:
            st.session_state.qa_chain = init_chatbot()
        except Exception as e:
            st.error(f"챗봇 초기화 중 오류가 발생했습니다: {str(e)}")
            st.stop()

# Sidebar for additional controls
with st.sidebar:
    st.title("보험 챗봇 설정")
    st.markdown("---")
    
    # Model selection
    model_name = st.selectbox(
        "모델 선택",
        ["llama2", "mistral", "gemma"],
        index=0
    )
    
    # Temperature control
    temperature = st.slider(
        "창의성",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="값이 높을수록 창의적인 답변을 생성합니다."
    )
    
    # Clear chat button
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 보험 챗봇")
st.caption("보험 약관에 대해 궁금한 점을 물어보세요.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("질문을 입력하세요"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                # Get response from QA chain
                response = st.session_state.qa_chain.run(prompt)
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
