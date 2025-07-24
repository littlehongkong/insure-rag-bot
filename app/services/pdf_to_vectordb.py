import os
import PyPDF2
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import supabase
from datetime import datetime

# Load environment variables
load_dotenv()

class PDFToVectorDB:
    def __init__(self, use_openai=False, openai_api_key=None):
        """
        PDF를 벡터 DB에 저장하는 클래스

        Args:
            use_openai (bool): OpenAI 임베딩 사용 여부 (False면 무료 모델 사용)
            openai_api_key (str): OpenAI API 키 (use_openai=True일 때 필요)
        """
        self.use_openai = use_openai

        if use_openai and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
        else:
            # 무료 임베딩 모델 사용 (sentence-transformers)
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.embeddings = None

        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # 로컬 ChromaDB 클라이언트 초기화
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'chroma_db')
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # PDF 저장 경로 설정
        self.pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pdfs')
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Supabase 클라이언트 초기화
        load_dotenv()
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)

    def download_pdf(self, summary_seq: str, product_info: Dict[str, Any]) -> Optional[str]:
        """PDF를 다운로드하고 저장된 경로를 반환합니다."""
        try:
            url = f"https://kpub.knia.or.kr/file/download/{summary_seq}.do"
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            
            # 파일명 생성 (TP_CODE_PCODE_타임스탬프.pdf)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{product_info.get('TP_CODE', '')}_{product_info.get('P_CODE', '')}_{timestamp}.pdf"
            filepath = os.path.join(self.pdf_dir, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return filepath
            
        except Exception as e:
            print(f"PDF 다운로드 실패 (SUMMARY_SEQ: {summary_seq}): {str(e)}")
            return None
            
    def fetch_insurance_products(self) -> List[Dict[str, Any]]:
        """insurance_products_raw 테이블에서 데이터를 조회합니다."""
        try:
            response = self.supabase.table('insurance_products_raw').select('*').execute()
            return response.data if hasattr(response, 'data') else []
        except Exception as e:
            print(f"데이터베이스 조회 실패: {str(e)}")
            return []
            
    def process_insurance_products(self, collection_name: str = "insurance_docs") -> None:
        """보험 상품 데이터를 처리하여 PDF를 다운로드하고 벡터 DB에 저장합니다."""
        products = self.fetch_insurance_products()
        if not products:
            print("처리할 보험 상품이 없습니다.")
            return
            
        for product in products:
            summary_seq = product.get('SUMMARY_SEQ')
            if not summary_seq:
                print(f"SUMMARY_SEQ가 없는 상품 건너뜁니다: {product.get('ID')}")
                continue
                
            print(f"처리 중: {product.get('TP_NAME')} - {product.get('P_CODE_NM')}")
            
            # PDF 다운로드
            pdf_path = self.download_pdf(summary_seq, product)
            if not pdf_path:
                print(f"PDF 다운로드 실패: {summary_seq}")
                continue
                
            # PDF에서 텍스트 추출 (메타데이터와 함께)
            try:
                documents = self.extract_text_from_pdf(pdf_path, product_info=product)
                if not documents:
                    print(f"텍스트 추출 실패: {pdf_path}")
                    continue
                
                # 벡터 DB에 저장
                self.save_to_chroma_manual(documents, collection_name)
                print(f"처리 완료: {pdf_path}")
                
            except Exception as e:
                print(f"처리 중 오류 발생 ({pdf_path}): {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path, product_info=None):
        """PDF에서 텍스트 추출
        
        Args:
            pdf_path (str): PDF 파일 경로
            product_info (dict, optional): 상품 정보가 포함된 딕셔너리. 기본값은 None입니다.
                - ID: 상품 ID
                - TP_NAME: 상품명
                - TP_CODE: 상품 코드
                - P_CODE_NM: 상품 코드명
        """
        try:
            # PyPDFLoader 사용 (권장)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 메타데이터 추가
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
            print(f"PyPDFLoader 오류, PyPDF2로 대체: {e}")
            # PyPDF2 사용 (백업)
            documents = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    metadata = {
                        "source": pdf_path, 
                        "page": page_num
                    }
                    
                    # 상품 정보가 있으면 메타데이터에 추가
                    if product_info:
                        metadata.update({
                            'id': str(product_info.get('ID')),
                            'tp_name': product_info.get('TP_NAME', ''),
                            'tp_code': product_info.get('TP_CODE', ''),
                            'p_code_nm': product_info.get('P_CODE_NM', '')
                        })
                        
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(doc)
            return documents

    def create_embeddings(self, texts):
        """텍스트를 임베딩으로 변환"""
        if self.use_openai:
            # OpenAI 임베딩 사용
            return [self.embeddings.embed_query(text) for text in texts]
        else:
            # 무료 모델 사용
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()

    def save_to_chroma_manual(self, documents, collection_name="pdf_collection"):
        """로컬 ChromaDB에 문서 저장 (무료 임베딩 모델 사용시)"""
        if not documents:
            print("저장할 문서가 없습니다.")
            return

        # 텍스트 분할
        texts = self.text_splitter.split_documents(documents)
        print(f"총 {len(texts)}개의 청크로 분할되었습니다.")

        # 텍스트와 메타데이터 준비
        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = self.create_embeddings(text_contents)

        # 로컬 ChromaDB에 저장
        try:
            # 컬렉션 생성 또는 가져오기
            try:
                collection = self.chroma_client.get_collection(collection_name)
                print(f"기존 컬렉션 '{collection_name}'에 저장 중...")
            except:
                collection = self.chroma_client.create_collection(name=collection_name)
                print(f"새 컬렉션 '{collection_name}'을 생성하고 저장 중...")

            # 배치 단위로 저장 (메모리 사용량 제한을 위해)
            batch_size = 50
            for i in range(0, len(text_contents), batch_size):
                batch_texts = text_contents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                # 배치별로 문서 추가
                collection.upsert(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"{min(i + batch_size, len(text_contents))}/{len(text_contents)} 문서 처리 완료")

            print(f"\n총 {len(text_contents)}개의 문서가 로컬 ChromaDB에 성공적으로 저장되었습니다.")
            print(f"저장 경로: {os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'chroma_db')}")
            return True

        except Exception as e:
            print(f"문서 저장 중 오류 발생: {str(e)}")
            return False

    def search_similar(self, query, collection_name="pdf_collection", k=5):
        """유사한 문서 검색"""
        try:
            collection = self.chroma_client.get_collection(collection_name)

            # 쿼리 임베딩 생성
            query_embedding = self.create_embeddings([query])[0]

            # 검색 수행
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            return results
        except Exception as e:
            print(f"검색 오류: {e}")
            return None

    def process_pdf(self, pdf_path, collection_name="pdf_collection"):
        """PDF 파일을 처리하여 벡터 DB에 저장"""
        print(f"PDF 처리 시작: {pdf_path}")

        # PDF에서 텍스트 추출
        documents = self.extract_text_from_pdf(pdf_path)
        print(f"추출된 페이지 수: {len(documents)}")

        # 벡터 DB에 저장
        if self.use_openai:
            collection = self.save_to_chroma_manual(documents, collection_name)
        else:
            collection = self.save_to_chroma_manual(documents, collection_name)

        print("PDF 처리 완료!")
        return collection


def process_pdf_directory(pdf_dir: str, processor, collection_name: str = "insurance_pdfs") -> None:
    """지정된 디렉토리의 모든 PDF 파일을 처리하여 벡터 DB에 저장"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"오류: '{pdf_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"'{pdf_dir}' 디렉토리에 PDF 파일이 없습니다.")
        return

    print(f"\n{'='*50}")
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
    print(f"{'='*50}\n")

    for pdf_file in pdf_files:
        try:
            print(f"\n[처리 중] {pdf_file.name}")
            print("-" * 50)

            # PDF에서 텍스트 추출
            documents = processor.extract_text_from_pdf(str(pdf_file))
            if not documents:
                print(f"경고: '{pdf_file}'에서 텍스트를 추출할 수 없습니다. 건너뜁니다.")
                continue

            # Chroma에 저장
            success = processor.save_to_chroma_manual(
                documents,
                collection_name=collection_name
            )

            if success:
                print(f"[완료] {pdf_file.name} 처리 완료")
            else:
                print(f"[실패] {pdf_file.name} 처리 중 오류 발생")

        except Exception as e:
            print(f"오류: {pdf_file} 처리 중 예외 발생 - {str(e)}")
            continue


def main():
    # 환경 변수에서 OpenAI API 키 로드
    openai_api_key = os.getenv("OPENAI_API_KEY")
    use_openai = False  # 기본값은 무료 모델 사용
    
    # PDFToVectorDB 인스턴스 생성
    print("PDF 처리기 초기화 중...")
    pdf_processor = PDFToVectorDB(use_openai=use_openai, openai_api_key=openai_api_key)
    
    try:
        # 1. 데이터베이스에서 보험 상품 정보를 가져와 PDF 다운로드 및 처리
        print("\n보험 상품 데이터베이스에서 PDF 다운로드 및 처리 중...")
        pdf_processor.process_insurance_products(collection_name="insurance_docs")
        print("보험 상품 PDF 처리 완료!")
    except Exception as e:
        print(f"보험 상품 처리 중 오류 발생: {str(e)}")
        print("PDF 디렉토리에서 파일 처리를 시도합니다...")
    
    # 2. PDF 디렉토리에서 직접 파일 처리 (백업)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "pdfs")
        
        # PDF 디렉토리가 존재하는지 확인
        if os.path.exists(pdf_dir) and os.path.isdir(pdf_dir):
            print(f"\nPDF 디렉토리에서 파일 처리 중: {pdf_dir}")
            process_pdf_directory(pdf_dir, pdf_processor, collection_name="insurance_pdfs")
        else:
            print(f"\nPDF 디렉토리를 찾을 수 없습니다: {pdf_dir}")
    except Exception as e:
        print(f"PDF 디렉토리 처리 중 오류 발생: {str(e)}")
    
    # 검색 예제 실행
    print("\n검색 예제:")
    search_example(pdf_processor, "보험 상품에 대한 정보", collection_name="insurance_pdfs")

    print("\n모든 처리가 완료되었습니다.")


def search_example(processor, query: str = "보험 상품에 대한 정보", collection_name: str = "insurance_pdfs", k: int = 3):
    """검색 예제 함수"""
    print(f"\n'{query}'에 대한 유사 문서 검색 중...")
    results = processor.search_similar(query, collection_name=collection_name, k=k)
    
    if results and 'documents' in results and results['documents'] and results['metadatas']:
        print(f"\n=== 검색 결과 (상위 {k}개) ===")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            source = metadata.get('source', '알 수 없음')
            print(f"\n{i}. 출처: {os.path.basename(source)}")
            print(f"   관련 상품: {metadata.get('tp_name', '알 수 없음')} ({metadata.get('tp_code', '')})")
            print(f"   페이지: {metadata.get('page', 0) + 1} 페이지")
            print("-" * 80)
            print(doc[:500] + ("..." if len(doc) > 500 else ""))
    else:
        print("검색 결과가 없거나 오류가 발생했습니다.")


if __name__ == "__main__":
    main()