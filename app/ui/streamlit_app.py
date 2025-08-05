import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import supabase

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Page configuration
st.set_page_config(
    page_title="보험 챗봇",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text: str) -> list[float]:
    """Generate embedding for the given text"""
    return embedding_model.encode(text).tolist()

from langchain.schema import BaseRetriever, Document

class SupabaseRetriever(BaseRetriever):
    def __init__(self, supabase_client, embedding_model, k: int = 3):
        super().__init__()
        self._supabase = supabase_client
        self._embedding_model = embedding_model
        self._k = k
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve relevant documents from Supabase using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = get_embedding(query)
            
            # Query Supabase
            response = self._supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_count': self._k
                }
            ).execute()
            
            if not hasattr(response, 'data') or not response.data:
                return []
            
            # Convert to LangChain Document format
            documents = []
            for doc in response.data:
                documents.append(
                    Document(
                        page_content=doc.get('content', ''),
                        metadata={
                            'source': doc.get('source', ''),
                            'page': doc.get('page', 0)
                        }
                    )
                )
            return documents
            
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def aget_relevant_documents(self, query: str) -> list[Document]:
        """Async version of get_relevant_documents"""
        return self.get_relevant_documents(query)

# Initialize chatbot
def init_chatbot():
    # Initialize retriever
    retriever = SupabaseRetriever(supabase_client, embedding_model, k=3)

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

    # Create QA chain with custom retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
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
