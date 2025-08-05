from typing import Optional, List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import supabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FreeInsuranceChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the Insurance Chatbot with free alternatives.

        Args:
            persist_directory (str): Path to the directory containing the Chroma DB
        """
        # 무료 임베딩 모델 (한국어 지원)
        self.embedding = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        # 무료 LLM (Ollama 사용)
        # 사전에 'ollama pull llama2' 또는 'ollama pull mistral' 실행 필요
        self.llm = Ollama(
            model="llama2",  # 또는 "mistral", "codellama" 등
            temperature=0
        )

        # 프롬프트 템플릿
        self.prompt_template = """
        당신은 보험 약관을 분석하는 전문가입니다.

        다음 내용을 참고하여 질문에 정확하게 답해주세요:

        [보장 항목 발췌]
        {context}

        [사용자 질문]
        {question}

        답변을 한국어로 간결하고 정확하게 제공해주세요.

        답변:
        """

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # QA 체인 초기화
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        """Query the chatbot with a question.

        Args:
            question (str): The user's question

        Returns:
            str: The chatbot's response
        """
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"질문 처리 중 오류가 발생했습니다: {str(e)}"


class SupabaseChatbot:
    def __init__(self):
        """Initialize the chatbot with Supabase connection."""
        # Initialize Supabase client
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key must be set in environment variables")
            
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)
        
        # Initialize LLM for generating responses
        self.llm = Ollama(model="llama2", temperature=0)

    def search_insurance_products(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for insurance products in insurance_products_raw table."""
        try:
            # Search in insurance_products_raw table directly
            response = self.supabase.table('insurance_products_raw')\
                .select('*')\
                .or_(f"TP_NAME.ilike.%{query}%, TP_ETC.ilike.%{query}%")\
                .limit(limit)\
                .execute()
            
            if hasattr(response, 'data'):
                return response.data
            return []
            
        except Exception as e:
            print(f"Error searching products: {str(e)}")
            return []
            
        except Exception as e:
            print(f"Error searching products: {str(e)}")
            return []

    def format_product_info(self, product: Dict[str, Any]) -> str:
        """Format product information into a readable string."""
        return f"""
        [보험 상품 정보]
        상품명: {product.get('TP_NAME', 'N/A')}
        보험사: {product.get('P_CODE_NM', 'N/A')}
        상품코드: {product.get('TP_CODE', 'N/A')}
        상품유형: {product.get('popup_category_sub', 'N/A')}
        
        [기본 정보]
        최소 보험료: {product.get('MIN_BILL', 'N/A')}원
        월 보험료: {product.get('TP_M_BILL', 'N/A')}원
        주 보험료: {product.get('TP_W_BILL', 'N/A')}원
        
        [기타 정보]
        치매보장금액: {product.get('DEMENTIA_AMT', 'N/A')}
        치매보장여부: {product.get('DEMENTIA_YN', 'N/A')}
        갱신여부: {product.get('RENEW_INS_SEQ', 'N/A')}
        
        [상세 정보]
        {product.get('TP_ETC', '추가 정보가 없습니다.')}
        """

    def query(self, question: str) -> str:
        """Generate a response to the user's question using Supabase data."""
        try:
            # Search for relevant products
            products = self.search_insurance_products(question)
            
            if not products:
                return "관련된 보험 상품을 찾을 수 없습니다. 다른 검색어로 시도해 주세요."
            
            # Format the response
            response = f"""
            검색하신 '{question}' 관련 보험 상품을 찾았습니다. 다음은 검색 결과입니다:
            ====================================
            """
            
            for i, product in enumerate(products[:3], 1):  # Show top 3 results
                response += f"\n{i}. {self.format_product_info(product)}"
            
            return response
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


# HuggingFace Transformers 로컬 실행 버전 (추천)
class HuggingFaceLocalChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """HuggingFace Transformers를 로컬에서 실행하는 버전"""
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        # 무료 임베딩
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        # 로컬 모델 파이프라인 설정
        model_name = "microsoft/DialoGPT-small"  # 작은 모델로 시작

        try:
            # GPU가 있으면 사용, 없으면 CPU
            device = 0 if torch.cuda.is_available() else -1

            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=device,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.1,
                pad_token_id=50256
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            # 대안: 더 작은 모델 사용
            pipe = pipeline(
                "text-generation",
                model="gpt2",
                max_new_tokens=100,
                do_sample=True,
                temperature=0.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        self.prompt_template = """
        Context: {context}

        Question: {question}

        Answer:
        """

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"오류 발생: {str(e)}"


# HuggingFace API 수정 버전 (Pydantic v2 호환)
class FixedHuggingFaceAPIChatbot:
    def __init__(self, persist_directory: str = "./chroma_db", hf_token: str = None):
        """수정된 HuggingFace API 버전"""
        from huggingface_hub import InferenceClient
        from langchain.llms.base import LLM
        from typing import Any, List, Optional
        from pydantic import Field

        # Pydantic v2 호환 커스텀 LLM 래퍼 클래스
        class HuggingFaceInferenceLLM(LLM):
            client: Any = Field(default=None)
            model_name: str = Field(default="")

            def __init__(self, model_name: str, token: str = None, **kwargs):
                super().__init__(**kwargs)
                self.model_name = model_name
                self.client = InferenceClient(model=model_name, token=token)

            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                try:
                    response = self.client.text_generation(
                        prompt,
                        max_new_tokens=200,
                        temperature=0.1,
                        return_full_text=False
                    )
                    return response
                except Exception as e:
                    return f"API 호출 실패: {str(e)}"

            @property
            def _llm_type(self) -> str:
                return "huggingface_inference"

        # 무료 임베딩
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

        # 수정된 HuggingFace LLM
        self.llm = HuggingFaceInferenceLLM(
            model_name="microsoft/DialoGPT-medium",
            token=hf_token
        )

        self.prompt_template = """
        보험 정보: {context}

        질문: {question}

        답변:
        """

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"오류 발생: {str(e)}"


# 더 간단한 대안: 순수 HuggingFace API 사용 (수정된 버전)
class SimpleHuggingFaceChatbot:
    def __init__(self, persist_directory: str = "./chroma_db", hf_token: str = None):
        """순수 HuggingFace API를 사용하는 간단한 버전"""
        import requests

        self.hf_token = hf_token
        # 작동하는 무료 모델들로 변경
        self.available_models = [
            "gpt2",  # 가장 안정적
            "microsoft/DialoGPT-small",  # 대화형
            "google/flan-t5-small",  # 텍스트 생성
            "facebook/blenderbot-400M-distill"  # 채팅봇
        ]
        self.current_model = "gpt2"  # 기본값
        self.api_url = f"https://api-inference.huggingface.co/models/{self.current_model}"
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        # 무료 임베딩
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def _call_hf_api(self, prompt: str) -> str:
        """HuggingFace API 직접 호출 (여러 모델 시도)"""
        import requests
        import time

        for model in self.available_models:
            api_url = f"https://api-inference.huggingface.co/models/{model}"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

            try:
                print(f"모델 시도 중: {model}")
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        if generated_text.strip():
                            self.current_model = model  # 성공한 모델로 업데이트
                            return generated_text.strip()
                    elif isinstance(result, dict) and 'generated_text' in result:
                        return result['generated_text'].strip()

                elif response.status_code == 503:
                    print(f"모델 {model} 로딩 중... 다음 모델 시도")
                    continue

                else:
                    print(f"모델 {model} 오류: {response.status_code} - {response.text}")
                    continue

            except Exception as e:
                print(f"모델 {model} 요청 실패: {str(e)}")
                continue

        return "죄송합니다. 현재 모든 모델이 사용 불가능합니다. 잠시 후 다시 시도해주세요."

    def query(self, question: str) -> str:
        try:
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content[:200] for doc in docs[:2]])  # 길이 제한

            # 간단한 프롬프트 구성
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

            # API 호출
            response = self._call_hf_api(prompt)
            return response

        except Exception as e:
            return f"오류 발생: {str(e)}"


# 완전 오프라인 버전 (인터넷 불필요)
class OfflineChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """완전 오프라인으로 작동하는 버전"""
        # 무료 임베딩
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def query(self, question: str) -> str:
        """단순 검색 기반 응답 (LLM 없이)"""
        try:
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)

            if not docs:
                return "관련 정보를 찾을 수 없습니다."

            # 가장 관련성 높은 문서 반환
            best_match = docs[0].page_content

            # 간단한 후처리
            lines = best_match.split('\n')
            relevant_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

            response = "관련 정보:\n" + "\n".join(relevant_lines[:3])
            return response

        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}"


# 로컬 GPT2 버전 (가장 안정적)
class LocalGPT2Chatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """로컬 GPT2를 사용하는 버전"""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

            print("GPT2 모델 로딩 중...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # 임베딩 및 벡터 DB
            self.embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            self.vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding
            )
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})

            print("모델 로딩 완료!")

        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            self.generator = None

    def query(self, question: str) -> str:
        if self.generator is None:
            return "모델이 로드되지 않았습니다."

        try:
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content[:150] for doc in docs[:2]])

            # 프롬프트 구성
            prompt = f"Insurance Information: {context}\n\nQuestion: {question}\n\nAnswer:"

            # 텍스트 생성
            result = self.generator(prompt, max_length=len(prompt.split()) + 50, num_return_sequences=1)

            # 생성된 텍스트에서 답변 부분만 추출
            generated = result[0]['generated_text']
            answer = generated[len(prompt):].strip()

            return answer if answer else "답변을 생성할 수 없습니다."

        except Exception as e:
            return f"오류 발생: {str(e)}"


# 사용 예시
if __name__ == "__main__":
    print("=== 사용 가능한 옵션들 ===")

    # 방법 1: Ollama 사용 (가장 안정적)
    # try:
    #     print("1. Ollama 버전 시도 중...")
    #     chatbot_ollama = FreeInsuranceChatbot()
    #     query = "입원비는 몇일 기준으로 보장되나요?"
    #     response = chatbot_ollama.query(query)
    #     print(f"질문: {query}")
    #     print(f"답변: {response}")
    # except Exception as e:
    #     print(f"Ollama 실패: {e}")
    #
    # # 방법 2: HuggingFace 로컬 실행 (추천)
    # try:
    #     print("\n2. HuggingFace 로컬 버전 시도 중...")
    #     chatbot_local = HuggingFaceLocalChatbot()
    #     query = "입원비는 몇일 기준으로 보장되나요?"
    #     response = chatbot_local.query(query)
    #     print(f"질문: {query}")
    #     print(f"답변: {response}")
    # except Exception as e:
    #     print(f"HuggingFace 로컬 실패: {e}")
    #
    # # 방법 3: 수정된 HuggingFace API
    # try:
    #     print("\n3. 수정된 HuggingFace API 버전 시도 중...")
    #     # HuggingFace 토큰이 필요합니다 (무료로 https://huggingface.co/settings/tokens에서 생성)
    #     chatbot_api = FixedHuggingFaceAPIChatbot(hf_token="your_hf_token_here")
    #     query = "입원비는 몇일 기준으로 보장되나요?"
    #     response = chatbot_api.query(query)
    #     print(f"질문: {query}")
    #     print(f"답변: {response}")
    # except Exception as e:
    #     print(f"HuggingFace API 실패: {e}")

    # 방법 4: 간단한 HuggingFace API (가장 안정적인 API 방식)
    try:
        print("\n4. 간단한 HuggingFace API 버전 시도 중...")
        chatbot_simple = SimpleHuggingFaceChatbot(hf_token=os.getenv("HF_TOKEN"))
        query = "입원비는 몇일 기준으로 보장되나요?"
        response = chatbot_simple.query(query)
        print(f"질문: {query}")
        print(f"답변: {response}")
    except Exception as e:
        print(f"간단한 API 실패: {e}")
