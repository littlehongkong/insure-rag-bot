import os
import re
import PyPDF2
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import supabase
from datetime import datetime
import numpy as np
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CloudRAGProcessor:
    def __init__(self, use_openai=False, openai_api_key=None, storage_type="supabase"):
        """
        클라우드 기반 RAG 시스템

        Args:
            use_openai (bool): OpenAI 임베딩 사용 여부
            openai_api_key (str): OpenAI API 키
            storage_type (str): "supabase", "postgresql", "pinecone" 중 선택
        """
        self.use_openai = use_openai
        self.storage_type = storage_type

        # 임베딩 모델 설정
        if use_openai and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
            self.embedding_dim = 1536  # OpenAI ada-002 차원
        else:
            # 무료 임베딩 모델 (한국어 지원)
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.embeddings = None
            self.embedding_dim = 384  # MiniLM 차원

        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # 스토리지 초기화
        self._init_storage()

    def _init_storage(self):
        """스토리지 초기화"""
        if self.storage_type == "supabase":
            self._init_supabase()
        elif self.storage_type == "postgresql":
            self._init_postgresql()
        elif self.storage_type == "pinecone":
            self._init_pinecone()

    def _init_supabase(self):
        """Supabase 초기화 (벡터 확장 사용)"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)

        # 벡터 테이블 생성 (최초 1회만)
        self._create_vector_table()

    def _init_postgresql(self):
        """PostgreSQL + pgvector 초기화 (Render 등)"""
        self.db_url = os.getenv('DATABASE_URL')  # Render에서 제공
        self.conn = psycopg2.connect(self.db_url)
        self._create_pgvector_table()

    def _init_pinecone(self):
        """Pinecone 초기화"""
        try:
            import pinecone
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment=os.getenv('PINECONE_ENV', 'us-west1-gcp-free')
            )

            index_name = "insurance-rag"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    index_name,
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
            self.pinecone_index = pinecone.Index(index_name)
        except ImportError:
            logger.error("pinecone-client 패키지가 필요합니다: pip install pinecone-client")

    def _create_vector_table(self):
        """Supabase에 벡터 테이블 생성"""
        try:
            # 벡터 확장 활성화 (Supabase 콘솔에서 미리 활성화 필요)
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_vectors (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({self.embedding_dim}),
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS document_vectors_embedding_idx 
            ON document_vectors USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """

            # RPC 함수로 실행 (Supabase에서 SQL 실행)
            logger.info("벡터 테이블 생성 완료 (수동으로 Supabase 콘솔에서 실행 필요)")

        except Exception as e:
            logger.warning(f"벡터 테이블 생성 실패: {e}")

    def _create_pgvector_table(self):
        """PostgreSQL에 벡터 테이블 생성"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_vectors (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_vectors_embedding_idx 
                    ON document_vectors USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                self.conn.commit()
                logger.info("PostgreSQL 벡터 테이블 생성 완료")
        except Exception as e:
            logger.error(f"PostgreSQL 테이블 생성 실패: {e}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩으로 변환"""
        if self.use_openai:
            return [self.embeddings.embed_query(text) for text in texts]
        else:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()



    def clean_text(self, text: str) -> str:
        # NULL 문자 및 기타 제어문자 제거
        return re.sub(r'[\x00-\x1F\x7F]', '', text)


    def save_documents_to_cloud(self, documents: List[Document]) -> bool:
        """문서를 클라우드 벡터 DB에 저장"""
        if not documents:
            logger.warning("저장할 문서가 없습니다.")
            return False

        # 텍스트 분할
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"총 {len(texts)}개의 청크로 분할되었습니다.")

        # 텍스트와 메타데이터 준비
        text_contents = [doc.page_content for doc in texts]

        metadatas = [doc.metadata for doc in texts]

        # 임베딩 생성
        logger.info("임베딩 생성 중...")
        embeddings = self.create_embeddings(text_contents)


        # 스토리지별 저장
        if self.storage_type == "supabase":
            return self._save_to_supabase(text_contents, embeddings, metadatas)
        elif self.storage_type == "postgresql":
            return self._save_to_postgresql(text_contents, embeddings, metadatas)
        elif self.storage_type == "pinecone":
            return self._save_to_pinecone(text_contents, embeddings, metadatas)

    def _save_to_supabase(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> bool:
        """Supabase에 저장"""
        try:
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_data = []
                for j in range(i, min(i + batch_size, len(texts))):
                    batch_data.append({
                        "content": self.clean_text(texts[j]),
                        "metadata": metadatas[j],
                        "embedding": embeddings[j]
                    })

                result = self.supabase.table('document_vectors').insert(batch_data).execute()
                logger.info(f"{min(i + batch_size, len(texts))}/{len(texts)} 문서 처리 완료")

            logger.info(f"총 {len(texts)}개 문서가 Supabase에 저장되었습니다.")
            return True

        except Exception as e:
            logger.error(f"Supabase 저장 오류: {e}")
            return False

    def _save_to_postgresql(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> bool:
        """PostgreSQL에 저장"""
        try:
            with self.conn.cursor() as cur:
                batch_size = 50
                for i in range(0, len(texts), batch_size):
                    batch_data = []
                    for j in range(i, min(i + batch_size, len(texts))):
                        batch_data.append((
                            texts[j],
                            json.dumps(metadatas[j]),
                            embeddings[j]
                        ))

                    cur.executemany("""
                        INSERT INTO document_vectors (content, metadata, embedding)
                        VALUES (%s, %s, %s)
                    """, batch_data)

                    self.conn.commit()
                    logger.info(f"{min(i + batch_size, len(texts))}/{len(texts)} 문서 처리 완료")

            logger.info(f"총 {len(texts)}개 문서가 PostgreSQL에 저장되었습니다.")
            return True

        except Exception as e:
            logger.error(f"PostgreSQL 저장 오류: {e}")
            return False

    def _save_to_pinecone(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> bool:
        """Pinecone에 저장"""
        try:
            vectors = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                vectors.append({
                    "id": f"doc_{i}_{datetime.now().timestamp()}",
                    "values": embedding,
                    "metadata": {**metadata, "content": text}
                })

            # 배치 업로드
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
                logger.info(f"{min(i + batch_size, len(vectors))}/{len(vectors)} 벡터 업로드 완료")

            logger.info(f"총 {len(vectors)}개 벡터가 Pinecone에 저장되었습니다.")
            return True

        except Exception as e:
            logger.error(f"Pinecone 저장 오류: {e}")
            return False

    def search_similar(self, query: str, k: int = 5) -> Dict[str, Any]:
        """유사 문서 검색"""
        query_embedding = self.create_embeddings([query])[0]

        if self.storage_type == "supabase":
            return self._search_supabase(query_embedding, k)
        elif self.storage_type == "postgresql":
            return self._search_postgresql(query_embedding, k)
        elif self.storage_type == "pinecone":
            return self._search_pinecone(query_embedding, k)

    def _search_supabase(self, query_embedding: List[float], k: int) -> Dict[str, Any]:
        """Supabase에서 검색"""
        try:
            # RPC 함수 사용 (Supabase 콘솔에서 미리 생성 필요)
            result = self.supabase.rpc('match_documents', {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,
                'match_count': k
            }).execute()

            return {
                'documents': [item['content'] for item in result.data],
                'metadatas': [item['metadata'] for item in result.data],
                'scores': [item['similarity'] for item in result.data]
            }
        except Exception as e:
            logger.error(f"Supabase 검색 오류: {e}")
            return {}

    def _search_postgresql(self, query_embedding: List[float], k: int) -> Dict[str, Any]:
        """PostgreSQL에서 검색"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT content, metadata, 1 - (embedding <=> %s) as similarity
                    FROM document_vectors
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (query_embedding, query_embedding, k))

                results = cur.fetchall()
                return {
                    'documents': [row[0] for row in results],
                    'metadatas': [row[1] for row in results],
                    'scores': [row[2] for row in results]
                }
        except Exception as e:
            logger.error(f"PostgreSQL 검색 오류: {e}")
            return {}

    def _search_pinecone(self, query_embedding: List[float], k: int) -> Dict[str, Any]:
        """Pinecone에서 검색"""
        try:
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )

            return {
                'documents': [match['metadata']['content'] for match in results['matches']],
                'metadatas': [match['metadata'] for match in results['matches']],
                'scores': [match['score'] for match in results['matches']]
            }
        except Exception as e:
            logger.error(f"Pinecone 검색 오류: {e}")
            return {}

    def extract_text_from_pdf(self, pdf_path: str, product_info: Dict = None) -> List[Document]:
        """PDF에서 텍스트 추출"""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            if product_info:
                for page in pages:
                    page.metadata.update({
                        'id': str(product_info.get('ID')),
                        'tp_name': product_info.get('TP_NAME', ''),
                        'tp_code': product_info.get('TP_CODE', ''),
                        'p_code_nm': product_info.get('P_CODE_NM', ''),
                        'source': pdf_path,
                        'page': page.metadata.get('page', 0)
                    })
            return pages

        except Exception as e:
            logger.error(f"PDF 추출 오류: {e}")
            return []

    def process_insurance_products(self) -> None:
        """보험 상품 PDF 처리 및 클라우드 저장"""
        # Supabase에서 보험 상품 정보 조회
        try:
            supabase_client = supabase.create_client(
                os.getenv('SUPABASE_URL'),
                os.getenv('SUPABASE_KEY')
            )
            # 개발용으로 10개만 처리 (디버깅 용이)
            # response = supabase_client.table('insurance_products_raw').select('*').limit(10).execute()
            response = supabase_client.table('insurance_products_raw').select('*').execute()
            products = response.data if hasattr(response, 'data') else []
            logger.info(f"총 {len(products)}개 보험 상품을 처리합니다.")
        except Exception as e:
            logger.error(f"데이터베이스 조회 실패: {e}")
            return

        successful_count = 0
        failed_count = 0

        for i, product in enumerate(products, 1):
            summary_seq = product.get('SUMMARY_SEQ')
            if not summary_seq:
                logger.warning(f"SUMMARY_SEQ가 없는 상품 건너뜀: {product.get('ID')}")
                failed_count += 1
                continue

            tp_name = product.get('TP_NAME', '알수없음')
            p_code_nm = product.get('P_CODE_NM', '알수없음')
            logger.info(f"[{i}/{len(products)}] 처리 중: {tp_name} - {p_code_nm}")

            # PDF 다운로드 (임시 파일로)
            pdf_path = self._download_pdf_temp(summary_seq, product)
            if not pdf_path:
                logger.error(f"PDF 다운로드 실패로 건너뜀: {summary_seq}")
                failed_count += 1
                continue

            # PDF 처리 및 클라우드 저장
            try:
                documents = self.extract_text_from_pdf(pdf_path, product_info=product)
                if documents:
                    success = self.save_documents_to_cloud(documents)
                    if success:
                        logger.info(f"✅ 처리 완료: {tp_name}")
                        successful_count += 1
                    else:
                        logger.error(f"❌ 벡터 저장 실패: {tp_name}")
                        failed_count += 1
                else:
                    logger.error(f"❌ 텍스트 추출 실패: {tp_name}")
                    failed_count += 1

                # 임시 파일 삭제 (안전하게)
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info(f"임시 파일 삭제: {pdf_path}")
                except Exception as delete_error:
                    logger.warning(f"임시 파일 삭제 실패: {delete_error}")

            except Exception as e:
                failed_count += 1
                logger.error(f"❌ 처리 중 오류 ({tp_name}): {e}")

                raise

        # 최종 결과 출력
        logger.info(f"\n{'=' * 50}")
        logger.info(f"처리 완료! 성공: {successful_count}개, 실패: {failed_count}개")
        logger.info(f"{'=' * 50}")

    def _download_pdf_temp(self, summary_seq: str, product_info: Dict) -> Optional[str]:
        """PDF를 임시 파일로 다운로드"""
        try:
            url = f"https://kpub.knia.or.kr/file/download/{summary_seq}.do"

            # SSL 경고 무시 설정
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            response = requests.get(url, stream=True, verify=False, timeout=30)
            response.raise_for_status()

            # Windows 호환 임시 디렉토리 생성
            import tempfile
            temp_dir = tempfile.gettempdir()  # Windows: C:\Users\USER\AppData\Local\Temp

            # 안전한 파일명 생성 (특수문자 제거)
            safe_tp_code = "".join(c for c in str(product_info.get('TP_CODE', 'unknown')) if c.isalnum())
            temp_filename = os.path.join(temp_dir, f"{safe_tp_code}_{summary_seq}.pdf")

            # 파일 저장
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 빈 청크 필터링
                        f.write(chunk)

            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                logger.info(f"PDF 다운로드 성공: {temp_filename} ({os.path.getsize(temp_filename)} bytes)")
                return temp_filename
            else:
                logger.error(f"PDF 파일이 비어있거나 생성되지 않음: {temp_filename}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"PDF 다운로드 타임아웃 (SUMMARY_SEQ: {summary_seq})")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"PDF 다운로드 네트워크 오류 (SUMMARY_SEQ: {summary_seq}): {e}")
            return None
        except Exception as e:
            logger.error(f"PDF 다운로드 실패 (SUMMARY_SEQ: {summary_seq}): {e}")
            return None



def main():
    """메인 실행 함수"""
    # 환경 변수 확인
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    for var in required_vars:
        if not os.getenv(var):
            logger.error(f"환경 변수 {var}가 설정되지 않았습니다.")
            return

    # RAG 프로세서 초기화
    logger.info("RAG 시스템 초기화 중...")
    rag_processor = CloudRAGProcessor(
        use_openai=False,  # 무료 모델 사용
        storage_type="supabase"  # "supabase", "postgresql", "pinecone" 중 선택
    )

    # 보험 상품 처리 (개발용으로 제한된 수량)
    logger.info("보험 상품 PDF 처리 중...")
    rag_processor.process_insurance_products()



if __name__ == "__main__":
    main()