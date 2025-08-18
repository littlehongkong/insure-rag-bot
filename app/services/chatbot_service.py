import time
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaRAGChatbot:
    def __init__(self,
                 db_dir: str = "chroma_db",
                 collection_name: str = "insurance_pdfs",
                 openai_api_key: Optional[str] = None,
                 default_model: str = "gpt-5-nano",  # 실제 존재하는 모델로 변경
                 temperature: float = 0.1):

        self.db_dir = str(Path(__file__).resolve().parent.parent.parent / db_dir)
        self.collection_name = collection_name
        self.default_model = default_model
        self.temperature = temperature

        # 컴포넌트 초기화
        self.embeddings = None
        self.db = None
        self.qa_chain = None
        self.conversational_chain = None
        self.memory = None
        self.llm = None

        # OpenAI API 키 설정
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 매개변수로 전달하세요.")

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        logger.info(f"ChromaRAG 챗봇 초기화 완료 - DB: {self.db_dir}, Collection: {self.collection_name}")

    @contextmanager
    def _timer(self, operation: str):
        """시간 측정 컨텍스트 매니저"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"[Timer] {operation}: {elapsed:.3f}초")

    def load_embeddings(self) -> None:
        """임베딩 모델 로딩"""
        if self.embeddings is not None:
            return

        with self._timer("임베딩 모델 로딩"):
            self.embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    def load_vectorstore(self) -> None:
        """벡터 스토어 로딩"""
        if self.db is not None:
            return

        if not self.embeddings:
            self.load_embeddings()

        with self._timer("벡터스토어 로딩"):
            try:
                self.db = Chroma(
                    persist_directory=self.db_dir,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )

                # 문서 수 확인
                docs = self.db.get()
                doc_count = len(docs.get('documents', []))
                logger.info(f"로딩된 문서 수: {doc_count}")

                if doc_count == 0:
                    logger.warning("벡터 스토어에 문서가 없습니다!")

            except Exception as e:
                logger.error(f"벡터스토어 로딩 실패: {e}")
                raise

    def load_llm(self, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        """LLM 모델 로딩"""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature

        # 이미 같은 모델이 로딩되어 있으면 스킵
        if self.llm and hasattr(self.llm, 'model_name') and self.llm.model_name == model:
            return

        with self._timer(f"OpenAI {model} 로딩"):
            try:
                self.llm = ChatOpenAI(
                    api_key=self.openai_api_key,
                    model=model,
                    temperature=temperature
                )
            except Exception as e:
                logger.error(f"LLM 로딩 실패: {e}")
                raise

    def _get_korean_prompt_template(self) -> str:
        """한국어 최적화 프롬프트 템플릿"""
        return """당신은 보험 전문가입니다. 다음 보험 관련 문서들을 참고하여 질문에 대해 정확하고 상세한 한국어 답변을 제공해주세요.

참고 문서:
{context}

질문: {question}

답변 지침:
1. 반드시 제공된 문서의 정보만을 근거로 답변하세요
2. 문서에 해당 정보가 없으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 명확히 표현하세요
3. 추측이나 일반적인 보험 지식으로 대답하지 마세요
4. 답변은 구체적이고 실용적으로 작성하세요
5. 보험 상품명, 코드, 조건 등은 정확히 인용하세요

답변:"""

    def build_chain(self) -> None:
        """QA 체인 구축"""
        if self.qa_chain is not None:
            return

        if not self.db:
            self.load_vectorstore()
        if not self.llm:
            self.load_llm()

        with self._timer("QA 체인 구축"):
            try:
                # 검색 파라미터 최적화 - k값 증가로 더 많은 컨텍스트 제공
                retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # 3 → 5로 증가
                )

                prompt = PromptTemplate(
                    template=self._get_korean_prompt_template(),
                    input_variables=["context", "question"]
                )

                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
            except Exception as e:
                logger.error(f"QA 체인 구축 실패: {e}")
                raise

    def build_conversational_chain(self, memory_window: int = 5) -> None:
        """대화형 체인 구축"""
        if self.conversational_chain is not None:
            return

        if not self.db:
            self.load_vectorstore()
        if not self.llm:
            self.load_llm()

        with self._timer("대화형 체인 구축"):
            try:
                # 메모리 설정
                self.memory = ConversationBufferWindowMemory(
                    k=memory_window,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key='answer'
                )

                # 검색기 설정 - 더 많은 문서 검색
                retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )

                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=False  # 로그 정리를 위해 False로 변경
                )
            except Exception as e:
                logger.error(f"대화형 체인 구축 실패: {e}")
                raise

    def _extract_metadata_info(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """메타데이터에서 상품 정보 추출"""
        return {
            "product_name": metadata.get("item_name", metadata.get("tp_name", "알 수 없는 상품")),
            "product_code": metadata.get("item_code", metadata.get("tp_code", "코드 없음")),
            "insurer_name": metadata.get("insurer_name", "보험사 정보 없음"),
            "page": str(metadata.get("page", 0) + 1),  # 0-based → 1-based
            "is_table": "표 데이터" if metadata.get("is_table", False) else "일반 텍스트"
        }

    def ask_with_detailed_context(self, question: str, k: int = 5, model: Optional[str] = None) -> Dict[str, Any]:
        """상세한 컨텍스트와 메타데이터를 포함한 질문 처리"""
        if not self.db:
            self.load_vectorstore()

        logger.info(f"질문 처리 시작: {question}")

        with self._timer("전체 처리"):
            # 1. 문서 검색
            with self._timer("문서 검색"):
                docs = self.db.similarity_search_with_score(question, k=k)

            if not docs:
                return {
                    "answer": "관련 문서를 찾을 수 없습니다.",
                    "sources": [],
                    "context_used": ""
                }

            # 2. 컨텍스트 구성 - 더 상세한 정보 포함
            context_parts = []
            source_info = []

            for i, (doc, score) in enumerate(docs, 1):
                content = doc.page_content.strip()
                if not content:
                    continue

                meta_info = self._extract_metadata_info(doc.metadata)

                # 컨텍스트에 더 자세한 메타정보 포함
                context_part = f"""[문서 {i}] 
상품명: {meta_info['product_name']}
상품코드: {meta_info['product_code']}  
보험사: {meta_info['insurer_name']}
페이지: {meta_info['page']}
유형: {meta_info['is_table']}
유사도: {score:.3f}

내용:
{content[:500]}"""  # 300자 → 500자로 증가

                context_parts.append(context_part)

                # 소스 정보 저장
                source_info.append({
                    "product_name": meta_info['product_name'],
                    "product_code": meta_info['product_code'],
                    "page": meta_info['page'],
                    "similarity_score": score,
                    "content_preview": content[:100]
                })

            context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

            # 3. LLM 호출
            if not self.llm or (model and hasattr(self.llm, 'model_name') and self.llm.model_name != model):
                self.load_llm(model=model)

            prompt = f"""당신은 보험 전문가입니다. 다음 문서들을 분석하여 사용자의 질문에 대해 정확하고 포괄적인 답변을 제공해주세요.

각 문서는 [문서 번호], 상품명, 상품코드, 보험사, 페이지, 유형, 유사도 정보가 포함되어 있습니다.

{context}

사용자 질문: {question}

답변 작성 지침:
1. 가장 관련성이 높은 문서(유사도가 높은)를 우선적으로 참조하세요
2. 여러 문서에서 보완적인 정보가 있다면 종합하여 답변하세요  
3. 상품명과 코드를 정확히 명시하세요
4. 문서에 없는 정보는 추측하지 마세요
5. 답변 구조: 요약 → 상세 설명 → 참고 문서 정보
6. 질문의 의도에 맞춰 구체적인 수치와 조건을 반드시 포함하세요.

전문가 답변:"""

            with self._timer("OpenAI API 호출"):
                try:
                    response = self.llm.invoke(prompt)
                    answer = response.content.strip()
                except Exception as e:
                    logger.error(f"OpenAI API 호출 실패: {e}")
                    return {
                        "answer": f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                        "sources": source_info,
                        "context_used": context
                    }

            logger.info(f"답변 생성 완료 - 길이: {len(answer)}자, 참조 문서: {len(source_info)}개")

            return {
                "answer": answer,
                "sources": source_info,
                "context_used": context
            }

    def ask_conversational(self, question: str, memory_window: int = 5) -> str:
        """대화 히스토리를 유지하는 질문 방식"""
        if not self.conversational_chain:
            self.build_conversational_chain(memory_window)

        logger.info(f"대화형 질문: {question}")

        with self._timer("대화형 처리"):
            try:
                result = self.conversational_chain.invoke({"question": question})
                answer = result.get("answer", "").strip()

                if not answer:
                    logger.warning("대화형 응답이 비어있음 - 상세 모드로 재시도")
                    detailed_result = self.ask_with_detailed_context(question)
                    return detailed_result["answer"]

                return answer

            except Exception as e:
                logger.error(f"대화형 체인 실행 오류: {e}")
                # 폴백으로 상세 모드 사용
                detailed_result = self.ask_with_detailed_context(question)
                return detailed_result["answer"]

    def ask(self, question: str) -> str:
        """기본 질문 처리 (하위 호환성)"""
        result = self.ask_with_detailed_context(question)
        return result["answer"]

    def clear_memory(self) -> None:
        """대화 메모리 초기화"""
        if self.memory:
            self.memory.clear()
            logger.info("대화 메모리 초기화 완료")

    def get_memory_messages(self) -> List:
        """현재 메모리에 저장된 메시지 반환"""
        if self.memory and hasattr(self.memory, 'chat_memory'):
            return self.memory.chat_memory.messages
        return []

    def test_connection(self) -> bool:
        """OpenAI 연결 테스트"""
        try:
            if not self.llm:
                self.load_llm()

            test_response = self.llm.invoke("안녕하세요. 간단한 테스트입니다.")
            logger.info(f"OpenAI 연결 성공: {test_response.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"OpenAI 연결 실패: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        if not self.db:
            self.load_vectorstore()

        try:
            collection_data = self.db.get()
            doc_count = len(collection_data.get('documents', []))

            # 샘플 메타데이터 분석
            sample_meta = {}
            if collection_data.get('metadatas'):
                sample_meta = collection_data['metadatas'][0] if collection_data['metadatas'] else {}

            return {
                "document_count": doc_count,
                "collection_name": self.collection_name,
                "sample_metadata_keys": list(sample_meta.keys()) if sample_meta else [],
                "db_path": self.db_dir
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {"error": str(e)}


# 사용 예시 및 테스트
if __name__ == "__main__":
    # API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        exit(1)

    # 챗봇 초기화
    chatbot = ChromaRAGChatbot(
        db_dir="chroma_db",  # 경로 수정 필요시 변경
        collection_name="insurance_pdfs",
        openai_api_key=openai_api_key,
        default_model="gpt-5-nano"  # 실제 존재하는 모델 사용
    )

    # 연결 테스트
    print("\n🔧 시스템 상태 확인 중...")
    if not chatbot.test_connection():
        print("❌ OpenAI 연결 실패")
        exit(1)

    # 컬렉션 정보 확인
    collection_info = chatbot.get_collection_info()
    print(f"📊 컬렉션 정보: {collection_info}")

    # 테스트 질문
    print("\n" + "=" * 60)
    print("🚀 개선된 RAG 챗봇 테스트")
    print("=" * 60)

    test_questions = [
        "무배당THE채우는335간편고지종신보험(해약환급금일부지급형)_체증형 보험의 특징을 상세히 설명해주세요.",
        "이 보험의 보장 내용과 가입 조건은 어떻게 되나요?",
        "해약환급금 일부지급형이 무엇인지 설명해주세요."
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 질문 {i}: {question}")
        print("-" * 50)

        # 상세 컨텍스트로 질문
        result = chatbot.ask_with_detailed_context(question)

        print(f"💡 답변:\n{result['answer']}")
        print(f"\n📚 참조 문서 수: {len(result['sources'])}")

        # 참조 문서 정보 출력
        for j, source in enumerate(result['sources'][:3], 1):  # 상위 3개만 표시
            print(f"   {j}. {source['product_name']} (코드: {source['product_code']}) "
                  f"- 유사도: {source['similarity_score']:.3f}")

    print(f"\n✅ 테스트 완료!")