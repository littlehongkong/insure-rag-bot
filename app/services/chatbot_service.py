# app/services/chatbot_service.py

import os
from supabase import create_client
from langchain.llms import Ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SupabaseChatbot:
    def __init__(self):
        # Supabase 설정
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Supabase URL과 Key가 필요합니다.")
        self.client = create_client(url, key)

        # ✅ 768차원 임베딩 모델 (all-mpnet-base-v2)
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Ollama LLM (llama2 등)
        self.llm = Ollama(model="llama2", temperature=0.2)

    def get_query_embedding(self, query: str) -> List[float]:
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def query_supabase(self, query: str, k: int = 3):
        query_vector = self.get_query_embedding(query)

        # ✅ Supabase 벡터 유사도 검색 (pgvector + cosine distance)
        response = self.client.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_count": k
        })

        if hasattr(response, "data"):
            return response.data
        return []

    def build_prompt(self, context: List[str], question: str) -> str:
        context_text = "\n\n".join([doc["content"][:500] for doc in context])
        prompt = f"""당신은 보험 약관을 설명하는 전문가입니다.

아래 내용을 참고하여 사용자의 질문에 정확하게 답해주세요.

[약관 내용 발췌]
{context_text}

[질문]
{question}

[답변]
"""
        return prompt

    def query(self, user_question: str) -> str:
        try:
            documents = self.query_supabase(user_question)

            if not documents:
                return "해당 질문에 대한 관련 정보를 찾을 수 없습니다."

            prompt = self.build_prompt(documents, user_question)
            answer = self.llm(prompt)
            return answer.strip()

        except Exception as e:
            return f"오류 발생: {str(e)}"
