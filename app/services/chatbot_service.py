import time
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class ChromaRAGChatbot:
    def __init__(self, db_dir: str = "chroma_db", collection_name: str = "insurance_pdfs", openai_api_key: str = None):
        self.db_dir = str(Path(__file__).resolve().parent.parent.parent / db_dir)
        self.collection_name = collection_name
        self.embeddings = None
        self.db = None
        self.qa_chain = None
        self.llm = None
        self.openai_api_key = openai_api_key

        # OpenAI API 키 설정
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 매개변수로 전달하세요.")

        print(f"[Init] OpenAI RAG Chatbot initialized")
        print(f"[Init] ChromaRAGChatbot initialized with DB dir: {self.db_dir}, collection: {self.collection_name}")

    def load_embeddings(self):
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"[Time] Embedding 모델 로딩 완료: {time.time() - start_time:.2f}초")

    def load_vectorstore(self):
        if not self.embeddings:
            self.load_embeddings()

        start_time = time.time()

        # ✅ 기본 설정으로 간단하게 초기화
        self.db = Chroma(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        # 문서 수 확인
        try:
            docs = self.db.get()
            print(f"총 문서 수: {len(docs['documents'])}")
            if docs['documents']:
                print("문서 샘플:", docs['documents'][:1])  # 샘플 수 줄임
        except Exception as e:
            print(f"문서 조회 오류: {e}")

        print(f"[Time] Chroma 벡터스토어 로딩 완료: {time.time() - start_time:.2f}초")

    def load_llm(self, model: str = "gpt-5-nano", temperature: float = 0.1):
        """✅ 실제 존재하는 모델 사용 및 파라미터 최적화"""
        start_time = time.time()

        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=model,
            max_tokens=4096  # 응답 길이 제한
        )

        print(f"[Time] OpenAI {model} 로딩 완료: {time.time() - start_time:.2f}초")

    def build_chain(self):
        if not self.db:
            self.load_vectorstore()
        if not self.llm:
            self.load_llm()

        start_time = time.time()

        # ✅ 검색 파라미터 최적화
        retriever = self.db.as_retriever(search_kwargs={"k": 3})

        # ✅ 한국어 최적화 프롬프트 추가
        prompt_template = """다음 보험 관련 문서들을 참고하여 질문에 대해 정확하고 간결한 한국어 답변을 제공해주세요.

문서 내용:
{context}

질문: {question}

답변 (핵심 내용만 간결하게):"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # ✅ stuff 방식 사용 (map_reduce보다 빠르고 안정적)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # map_reduce → stuff로 변경
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        print(f"[Time] QA 체인 빌드 완료: {time.time() - start_time:.2f}초")

    def ask(self, question: str) -> str:
        """✅ 디버깅 정보 추가 및 에러 처리 강화"""
        if not self.qa_chain:
            self.build_chain()

        print(f"[Query] 질문 입력됨: {question}")

        # 관련 문서 검색
        try:
            retrieved_docs = self.qa_chain.retriever.get_relevant_documents(question)
            print(f"검색된 문서 수: {len(retrieved_docs)}")
        except Exception as e:
            print(f"문서 검색 오류: {e}")

        start_time = time.time()
        try:
            result = self.qa_chain.invoke({"query": question})  # invoke 사용
            print(f"[Time] 질문 응답 처리 완료: {time.time() - start_time:.2f}초")

            # ✅ 디버깅 정보 출력
            print(f"[Debug] 결과 키들: {list(result.keys())}")
            print(f"[Debug] result 길이: {len(result.get('result', ''))}")

            if not result.get('result', '').strip():
                print("⚠️ OpenAI 응답이 비어있습니다. ask_direct() 방식을 시도해보세요.")
                return self.ask_direct(question)

            return result["result"]

        except Exception as e:
            print(f"❌ QA 체인 실행 오류: {e}")
            print(f"❌ ask_direct() 방식으로 재시도합니다.")
            return self.ask_direct(question)

    def ask_direct(self, question: str, model: str = "gpt-5-nano") -> str:
        """✅ 직접 호출 방식 - 더 안정적이고 빠름"""
        if not self.db:
            self.load_vectorstore()

        print(f"\n⚡ [Direct Mode] {question}")
        total_start = time.time()

        # 1단계: 문서 검색
        search_start = time.time()
        docs = self.db.similarity_search(question, k=3)
        search_time = time.time() - search_start
        print(f"📄 검색 완료: {search_time:.3f}초 ({len(docs)}개 문서)")

        if not docs:
            return "관련 문서를 찾을 수 없습니다."

        # 2단계: 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:300].strip()
            meta = doc.metadata

            # 상품명, 코드 등 필요한 정보 가져오기 (없으면 기본값 처리)
            product_name = meta.get("tp_name", "알 수 없음 상품")
            product_code = meta.get("tp_code", "코드 없음")

            if content:
                context_parts.append(
                    f"[문서{i} | 상품명: {product_name} | 코드: {product_code}]\n{content}"
                )

        context = "\n\n".join(context_parts)
        print(f"📝 컨텍스트 길이: {len(context)}자")

        # 3단계: OpenAI API 직접 호출
        try:
            if not self.llm or self.llm.model_name != model:
                self.load_llm(model=model)

            prompt = f"""
            다음은 서로 다른 보험 상품에 대한 설명입니다.
            각 문서의 [상품명] 정보를 참고하여 사용자의 질문에 가장 적합한 상품의 내용을 기반으로 정확하고 간결하게 답변해주세요.

            {context}

            사용자 질문: {question}

            보험 전문가의 답변:"""

            api_start = time.time()
            response = self.llm.invoke(prompt)
            api_time = time.time() - api_start
            total_time = time.time() - total_start

            answer = response.content.strip()

            print(f"🤖 OpenAI API: {api_time:.3f}초")
            print(f"🎯 전체 시간: {total_time:.3f}초")
            print(f"📝 답변 길이: {len(answer)}자")

            if not answer:
                return "OpenAI로부터 응답을 받지 못했습니다."

            return answer

        except Exception as e:
            print(f"❌ Direct API 호출 실패: {e}")
            return f"응답 생성 실패: {str(e)}"

    def ask_gpt4(self, question: str) -> str:
        """GPT-4를 사용한 고품질 응답"""
        return self.ask_direct(question, model="gpt-4")

    def test_connection(self):
        """✅ OpenAI 연결 테스트"""
        try:
            if not self.llm:
                self.load_llm()

            test_response = self.llm.invoke("안녕하세요. 간단한 테스트입니다.")
            print(f"✅ OpenAI 연결 성공: {test_response.content[:50]}...")
            return True
        except Exception as e:
            print(f"❌ OpenAI 연결 실패: {e}")
            return False


# 예시 실행
if __name__ == "__main__":

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        exit(1)

    chatbot = ChromaRAGChatbot(
        db_dir="/Users/benjamin/PycharmProjects/insure-rag-bot/chroma_db",
        collection_name="insurance_pdfs",
        openai_api_key=openai_api_key
    )

    # ✅ 연결 테스트
    print("\n🔧 OpenAI 연결 테스트 중...")
    if not chatbot.test_connection():
        print("❌ OpenAI 연결에 실패했습니다. API 키를 확인해주세요.")
        exit(1)

    # ✅ 질문 테스트
    print("\n" + "=" * 60)
    print("🚀 RAG 질의응답 테스트")
    print("=" * 60)

    question = "어린이 실손보험의 보장 범위는 어떻게 되나요?"

    # print("\n1️⃣ 기본 방식 (RetrievalQA):")
    # response1 = chatbot.ask(question)
    # print(f"\n[Answer] {response1}")

    print(f"\n2️⃣ 직접 방식 (더 빠름):")
    response2 = chatbot.ask_direct(question)
    print(f"\n[Direct Answer] {response2}")

    print(f"\n✅ 테스트 완료!")