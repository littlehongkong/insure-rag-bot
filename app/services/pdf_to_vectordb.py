import os
import re
import uuid
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from PyPDF2.errors import PdfReadError

import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import supabase

# 환경 변수 로드
load_dotenv()

class PDFToVectorDB:
    def __init__(self, use_openai=False, openai_api_key=None):
        self.use_openai = use_openai

        self.embedding_model = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
        self.embeddings = None
        tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )

        # 저장 경로 설정
        base_path = Path(__file__).resolve().parent.parent.parent
        self.pdf_dir = base_path / 'pdfs'
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB 클라이언트
        db_path = base_path / 'chroma_db'
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

        # Supabase 클라이언트 초기화
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)

    def search_similar(self, query: str, collection_name: str = "insurance_pdfs", k: int = 3):
        collection = self.chroma_client.get_collection(collection_name)
        query_embedding = self.create_embeddings([query])[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        return results

    def download_pdf(self, summary_seq: int, product_info: Dict[str, Any]) -> Optional[str]:
        try:
            url = f"https://kpub.knia.or.kr/file/download/{summary_seq}.do"
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{product_info.get('TP_CODE', '')}_{product_info.get('P_CODE', '')}_{timestamp}.pdf"
            filepath = self.pdf_dir / filename

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # === PDF 유효성 검사 ===
            with open(filepath, 'rb') as f:
                try:
                    reader = PyPDF2.PdfReader(f)
                    _ = len(reader.pages)  # 페이지 접근 시 오류 발생하면 invalid
                except Exception as e:
                    print(f"[오류] 유효하지 않은 PDF 파일입니다. 삭제합니다. ({filepath}): {e}")
                    filepath.unlink(missing_ok=True)
                    return None

            return str(filepath)

        except Exception as e:
            print(f"PDF 다운로드 실패 (SUMMARY_SEQ: {summary_seq}): {str(e)}")
            return None
    def clean_text(self, text: str) -> str:
        return re.sub(r'[\x00\u0000]', '', text)

    def extract_text_from_pdf(self, pdf_path, product_info=None):
        documents = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                metadata = {
                    "source": pdf_path,
                    "page": page_num
                }

                if product_info:
                    metadata.update({
                        'id': str(product_info.get('id', '')),
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
        if self.use_openai:
            return [self.embeddings.embed_query(text) for text in texts]
        else:
            embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
            return embeddings

    def save_to_chroma_manual(self, documents, collection_name="pdf_collection"):
        if not documents:
            print("저장할 문서가 없습니다.")
            return False

        texts = self.text_splitter.split_documents(documents)
        print(f"총 {len(texts)}개의 청크로 분할되었습니다.")

        text_contents = [doc.page_content for doc in texts]
        metadatas = [doc.metadata for doc in texts]
        ids = [f"doc_{uuid.uuid4()}" for _ in range(len(texts))]

        print("임베딩 생성 중...")
        embeddings = self.create_embeddings(text_contents)

        try:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                print(f"기존 컬렉션 '{collection_name}'에 저장 중...")
            except:
                collection = self.chroma_client.create_collection(name=collection_name)
                print(f"새 컬렉션 '{collection_name}'을 생성하고 저장 중...")

            batch_size = 50
            for i in range(0, len(texts), batch_size):
                collection.upsert(
                    documents=text_contents[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
                print(f"{min(i + batch_size, len(texts))}/{len(texts)} 문서 저장 완료")

            print(f"\n총 {len(texts)}개의 문서가 ChromaDB에 저장되었습니다.")
            return True

        except Exception as e:
            print(f"Chroma 저장 중 오류 발생: {str(e)}")
            return False

    def process_insurance_products(self, collection_name="insurance_pdfs") -> None:
        """Supabase에서 보험 상품 조회 후 PDF 다운로드, 텍스트 추출, 벡터 DB 저장"""
        try:
            # Supabase에서 모든 보험 상품 데이터 조회
            response = self.supabase.table('insurance_products_raw').select('*').execute()
            products = response.data if hasattr(response, 'data') else []
            print(f"처리할 보험 상품 개수: {len(products)}")
            if not products:
                print("처리할 보험 상품이 없습니다.")
                return

            for index, product in enumerate(products):
                print(f"처리 중: {index + 1}/{len(products)}")

                summary_seq = product.get('SUMMARY_SEQ')
                if not summary_seq:
                    print(f"SUMMARY_SEQ가 없는 상품 건너뜁니다: {product.get('id', '')}")
                    continue

                print(f"처리 중: {product.get('TP_NAME')} - {product.get('P_CODE_NM')}")

                # PDF 다운로드
                pdf_path = self.download_pdf(summary_seq, product)
                if not pdf_path:
                    print(f"PDF 다운로드 실패: SUMMARY_SEQ={summary_seq}")
                    continue

                # PDF에서 텍스트 추출
                documents = self.extract_text_from_pdf(pdf_path, product_info=product)
                if not documents:
                    print(f"텍스트 추출 실패: {pdf_path}")
                    continue

                # ChromaDB에 저장
                success = self.save_to_chroma_manual(documents, collection_name)
                if success:
                    print(f"처리 완료: {pdf_path}")
                else:
                    print(f"저장 실패: {pdf_path}")

        except Exception as e:
            print(f"보험 상품 처리 중 오류 발생: {str(e)}")

    def process_pdf(self, pdf_path, collection_name="pdf_collection"):
        print(f"PDF 처리 시작: {pdf_path}")
        documents = self.extract_text_from_pdf(pdf_path)
        print(f"추출된 페이지 수: {len(documents)}")

        success = self.save_to_chroma_manual(documents, collection_name=collection_name)
        if success:
            print("PDF 처리 완료!")
        return success



def process_pdf_directory(pdf_dir: str, processor: PDFToVectorDB, collection_name: str = "insurance_pdfs") -> None:
    """디렉토리 내 모든 PDF 파일을 처리하여 벡터 DB에 저장"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"오류: '{pdf_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"'{pdf_dir}' 디렉토리에 PDF 파일이 없습니다.")
        return

    print(f"\n총 {len(pdf_files)}개의 PDF 파일을 처리합니다.\n")

    for pdf_file in pdf_files:
        try:
            print(f"\n[처리 중] {pdf_file.name}")
            documents = processor.extract_text_from_pdf(str(pdf_file))
            if not documents:
                print(f"경고: '{pdf_file}'에서 텍스트를 추출할 수 없습니다. 건너뜁니다.")
                continue

            success = processor.save_to_chroma_manual(documents, collection_name=collection_name)
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
        pdf_processor.process_insurance_products(collection_name="insurance_pdfs")
        print("보험 상품 PDF 처리 완료!")
    except Exception as e:
        print(f"보험 상품 처리 중 오류 발생: {str(e)}")
        print("PDF 디렉토리에서 파일 처리를 시도합니다...")

    # 2. PDF 디렉토리에서 직접 파일 처리 (백업)
    # try:
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     pdf_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "pdfs")
    #
    #     # PDF 디렉토리가 존재하는지 확인
    #     if os.path.exists(pdf_dir) and os.path.isdir(pdf_dir):
    #         print(f"\nPDF 디렉토리에서 파일 처리 중: {pdf_dir}")
    #         process_pdf_directory(pdf_dir, pdf_processor, collection_name="insurance_pdfs")
    #     else:
    #         print(f"\nPDF 디렉토리를 찾을 수 없습니다: {pdf_dir}")
    # except Exception as e:
    #     print(f"PDF 디렉토리 처리 중 오류 발생: {str(e)}")

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