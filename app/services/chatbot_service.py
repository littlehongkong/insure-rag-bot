import os
from typing import List, Dict, Any
from supabase import create_client, Client
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class SupabaseRAGChatbot:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("환경 변수에 SUPABASE_URL 및 SUPABASE_KEY 설정 필요")

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        # 임베딩 모델 (한국어 지원)
        self.embedding = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Supabase의 document_vectors 테이블을 LangChain VectorStore로 연결
        self.vectorstore = SupabaseVectorStore(
            client=self.client,
            table_name="document_vectors",
            embedding=self.embedding
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM (Ollama)
        self.llm = Ollama(model="llama2", temperature=0)

        # 프롬프트 템플릿
        prompt_template = """
        당신은 보험 약관 전문가입니다.

        다음 문서를 참고하여 질문에 답변해주세요:

        {context}

        질문: {question}

        답변:
        """
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # RetrievalQA 체인 구성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        try:
            answer = self.qa_chain.run(question)
            return answer.strip()
        except Exception as e:
            return f"오류 발생: {e}"
