#B00312010_1_P
import os
import re
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import supabase  # pip install supabase

# =========================
# 환경 변수 로드
# =========================
load_dotenv()


class PDFToVectorDB:
    """
    보험 약관 PDF -> 텍스트/표 추출 -> 청크/임베딩 -> ChromaDB 저장 파이프라인
    - Supabase의 insurance_products 테이블을 기준으로 오늘(updated_at) 변경된 약관만 처리
    - 처리 전 같은 (insurer_code, item_code) 기존 벡터는 전량 삭제하여 최신 약관만 유지
    """

    def __init__(self, insurer_code: str = "LINA", collection_name: str = "insurance_pdfs", use_openai: bool = False, openai_api_key: Optional[str] = None):
        self.insurer_code = insurer_code
        self.collection_name = collection_name

        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.embeddings = None  # (옵션) OpenAI 쓸 때 할당

        # 한국어 임베딩 모델 (384차원)
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')

        # 길이 기반 텍스트 분할기 (토크나이저 기준)
        self.length_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
            separators=["\n\n", "\n", " ", ""]
        )

        # 저장 경로
        base_path = Path(__file__).resolve().parent.parent.parent
        self.base_pdf_dir = base_path / 'pdfs'
        self.base_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir = self.base_pdf_dir / self.insurer_code
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB (영구형)
        db_path = base_path / 'chroma_db'
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)

        # Supabase
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)

    # =========================
    # 유틸
    # =========================
    def _start_of_today_kst_iso(self) -> str:
        """KST(UTC+9) 오늘 00:00:00을 ISO8601로 반환"""
        kst = timezone(timedelta(hours=9))
        now_kst = datetime.now(kst)
        start = datetime(year=now_kst.year, month=now_kst.month, day=now_kst.day, tzinfo=kst)
        return start.isoformat()

    def _deterministic_id(self, insurer_code: str, item_code: str, file_name: str, page: int, chunk_idx: int) -> str:
        raw = f"{insurer_code}|{item_code}|{file_name}|{page}|{chunk_idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _clean_text(self, text: str) -> str:
        return re.sub(r'[\x00\u0000]', '', text)

    def _regex_chunk_split(self, text: str) -> List[str]:
        """
        표/리스트를 하나의 chunk로 묶기 위한 regex 분리 (원 코드 개선/재사용)
        """
        pattern = r"(\n\s*(?:[-•*]|\d+\.)\s.*|\n.*\t.*)"
        parts = re.split(pattern, text)
        chunks = []
        buffer = ""
        for part in parts:
            if not part:
                continue
            if re.match(pattern, part):
                buffer += part
            else:
                if buffer:
                    chunks.append(buffer.strip())
                    buffer = ""
                chunks.append(part.strip())
        if buffer:
            chunks.append(buffer.strip())
        return [c for c in chunks if c]

    # =========================
    # Supabase 조회
    # =========================
    def fetch_products_updated_today(self) -> List[Dict[str, Any]]:
        """
        오늘(KST) 업데이트된 보험 상품(약관)만 조회
        스키마: insurance_products
        """
        start_kst = self._start_of_today_kst_iso()
        res = self.supabase.table('insurance_products') \
            .select('*') \
            .eq('insurer_code', self.insurer_code) \
            .order('updated_at', desc=False) \
            .execute()

        # .gte('updated_at', start_kst) \
        return res.data if hasattr(res, 'data') else []

    # =========================
    # PDF 다운로드
    # =========================
    def download_pdf_from_dbrow(self, row: Dict[str, Any]) -> Optional[str]:
        """
        DB 레코드의 file_url / file_name 기준으로 PDF 다운로드
        """
        file_url = row.get('file_url')
        file_name = row.get('file_name') or f"{row.get('item_code', 'unknown')}.pdf"
        if not file_url:
            print(f"[경고] file_url 이 없어 다운로드 건너뜀: item_code={row.get('item_code')}")
            return None

        local_path = self.pdf_dir / file_name
        try:
            resp = requests.get(file_url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            # 유효성
            with open(local_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                _ = len(reader.pages)
            print(f"[다운로드 완료] {local_path}")
            return str(local_path)
        except Exception as e:
            print(f"[오류] PDF 다운로드 실패: {file_url} ({e})")
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass
            return None

    # =========================
    # 텍스트 / 표 추출
    # =========================
    def _build_metadata(self, base_meta: Dict[str, Any], pdf_path: str, page_num: int, is_table: bool) -> Dict[
        str, Any]:
        """메타데이터 생성 - 페이지 번호는 0-based에서 1-based로 변환"""
        md = {
            # 파일/페이지 (사용자 친화적으로 1-based 페이지 번호)
            "source": str(pdf_path),
            "page": page_num,  # 내부적으로는 0-based 유지
            "page_display": page_num + 1,  # 사용자에게 보여줄 때는 1-based
            "is_table": is_table,

            # DB 기본 메타
            "insurer_code": base_meta.get("insurer_code"),
            "insurer_name": base_meta.get("insurer_name"),
            "item_type": base_meta.get("item_type"),
            "category_name": base_meta.get("category_name"),
            "item_name": base_meta.get("item_name"),
            "item_code": base_meta.get("item_code"),
            "file_name": base_meta.get("file_name"),
            "file_url": base_meta.get("file_url"),
            "status": base_meta.get("status"),

            # details 확장
            "sell_open_date": (base_meta.get("details") or {}).get("sell_open_date"),
            "sell_end_date": (base_meta.get("details") or {}).get("sell_end_date"),
            "klia_product_code": (base_meta.get("details") or {}).get("klia_product_code"),
            "kcis_insurance_code": (base_meta.get("details") or {}).get("kcis_insurance_code"),
            "prod_pban_grp_cd": (base_meta.get("details") or {}).get("prod_pban_grp_cd"),
            "product_summary": (base_meta.get("details") or {}).get("product_summary"),
            "product_method": (base_meta.get("details") or {}).get("product_method"),
            "item_sequence": (base_meta.get("details") or {}).get("item_sequence"),
            "item_section": (base_meta.get("details") or {}).get("item_section"),
            "crawled_at": (base_meta.get("details") or {}).get("crawled_at"),
            "api_version": (base_meta.get("details") or {}).get("api_version"),

            # 인덱싱 시점
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }
        return md

    def extract_documents(self, pdf_path: str, db_row: Dict[str, Any]) -> List[Document]:
        import fitz  # PyMuPDF
        import pdfplumber

        docs: List[Document] = []

        # PyMuPDF로 페이지별 텍스트 추출 (정확한 페이지 매핑)
        try:
            fitz_doc = fitz.open(pdf_path)

            for page_num, page in enumerate(fitz_doc):
                page_text = page.get_text("text").strip()

                if page_text:
                    md = self._build_metadata(db_row, pdf_path, page_num, is_table=False)
                    docs.append(Document(
                        page_content=self._clean_text(page_text),
                        metadata=md
                    ))

            fitz_doc.close()
            print(f"[추출] PyMuPDF로 {len(docs)}개 페이지 처리")

        except Exception as e:
            print(f"[ERROR] PyMuPDF 텍스트 추출 실패: {e}")
            return []

        # 테이블 데이터는 별도로 pdfplumber로 추출 (페이지별 정확 매핑)
        try:
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_num, page in enumerate(plumber_pdf.pages):
                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables or []):
                        if not table or not any(any(cell for cell in row if cell) for row in table):
                            continue

                        # 테이블을 텍스트로 변환
                        table_rows = []
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            if any(cleaned_row):  # 빈 행 제외
                                table_rows.append(" | ".join(cleaned_row))

                        if table_rows:
                            table_text = f"[TABLE {table_idx + 1}]\n" + "\n".join(table_rows)
                            md = self._build_metadata(db_row, pdf_path, page_num, is_table=True)
                            # 테이블 메타데이터에 테이블 번호 추가
                            md["table_index"] = table_idx

                            docs.append(Document(
                                page_content=self._clean_text(table_text),
                                metadata=md
                            ))

            table_count = sum(1 for doc in docs if doc.metadata.get("is_table"))
            print(f"[추출] pdfplumber로 {table_count}개 테이블 처리")

        except Exception as e:
            print(f"[WARN] 테이블 추출 실패: {pdf_path} → {e}")

        return docs


    # =========================
    # 임베딩
    # =========================
    def create_embeddings(self, texts: List[str]):
        if self.use_openai:
            # 필요 시 OpenAI 임베딩으로 교체 (현재는 sentence_transformers 사용)
            return [self.embeddings.embed_query(t) for t in texts]
        return self.embedding_model.encode(texts, normalize_embeddings=True)

    # =========================
    # 저장(Chroma) — 기존 벡터 삭제 후 업서트
    # =========================
    def upsert_documents(self, docs: List[Document], insurer_code: str, item_code: str, file_name: str):
        if not docs:
            print("[경고] 저장할 문서가 없습니다.")
            return False

        # 1) 이전 데이터 삭제: 같은 (insurer_code, item_code) 전량 제거 → 개정 전 벡터 제거
        try:
            self.collection.delete(where={"insurer_code": insurer_code, "item_code": item_code})
            print(f"[정리] 기존 벡터 삭제 완료: insurer_code={insurer_code}, item_code={item_code}")
        except Exception as e:
            print(f"[경고] 기존 벡터 삭제 중 오류 (무시하고 진행): {e}")

        # 2) 청크화
        chunks: List[Document] = []
        for d in docs:
            # 표/리스트를 우선 블록 기준으로 분할
            blocks = self._regex_chunk_split(d.page_content)
            for block in blocks:
                # 길이 기반 세분화
                sub_texts = self.length_splitter.split_text(block)
                for sub in sub_texts:
                    chunks.append(Document(page_content=sub, metadata=d.metadata))

        print(f"[분할] 총 {len(chunks)}개 청크")

        # 3) 임베딩 및 업서트 (결정적 ID)
        texts = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        embeddings = self.create_embeddings(texts)

        ids = []
        per_page_counters = {}  # 페이지별 청크 인덱스
        for m in metas:
            page = int(m.get("page", 0))
            key = (page,)
            per_page_counters[key] = per_page_counters.get(key, 0) + 1
            chunk_idx = per_page_counters[key] - 1
            # 메타데이터에 chunk_index 추가
            m['chunk_index'] = chunk_idx
            ids.append(self._deterministic_id(insurer_code, item_code, file_name, page, chunk_idx))

        # 배치 업서트
        BATCH = 64
        try:
            for i in range(0, len(chunks), BATCH):
                self.collection.upsert(
                    documents=texts[i:i+BATCH],
                    embeddings=embeddings[i:i+BATCH],
                    metadatas=metas[i:i+BATCH],
                    ids=ids[i:i+BATCH]
                )
                print(f"[업서트] {min(i+BATCH, len(chunks))}/{len(chunks)}")
            print(f"[완료] 총 {len(chunks)}개 청크 업서트 완료")
            return True
        except Exception as e:
            print(f"[오류] Chroma 업서트 실패: {e}")
            return False

    # =========================
    # 메인 처리 루틴
    # =========================
    def index_today_updated_products(self):
        """
        오늘 변경된 약관만 색인
        """
        rows = self.fetch_products_updated_today()
        print(f"[조회] 오늘 변경된 약관: {len(rows)}건")
        if not rows:
            return

        for idx, row in enumerate(rows, 1):
            try:
                print(f"\n[{idx}/{len(rows)}] {row.get('item_name')} ({row.get('item_code')}) - {row.get('file_name')}")
                pdf_path = self.pdf_dir / row.get('file_name')
                # pdf_path = self.download_pdf_from_dbrow(row)
                # if not pdf_path:
                #     print("[건너뜀] PDF 다운로드 실패")
                #     continue

                docs = self.extract_documents(pdf_path, db_row=row)
                if not docs:
                    print("[건너뜀] 텍스트 추출 실패/없음")
                    continue

                ok = self.upsert_documents(
                    docs,
                    insurer_code=row.get("insurer_code"),
                    item_code=row.get("item_code"),
                    file_name=row.get("file_name")
                )
                if ok:
                    print(f"[색인 완료] {row.get('item_code')} / {row.get('file_name')}")
                else:
                    print(f"[실패] 색인 실패: {row.get('item_code')} / {row.get('file_name')}")
            except Exception as e:
                print(f"[예외] 처리 중 오류: {e}")

    def search_with_context(self, query: str, k: int = 5,
                            context_window: int = 1) -> List[Dict[str, Any]]:
        """
        컨텍스트 윈도우를 포함한 검색
        - 검색된 청크의 앞뒤 청크도 함께 반환하여 문맥 이해 향상
        """
        base_results = self.search(query, k=k)

        if not base_results.get("documents") or not base_results["documents"][0]:
            return []

        enriched_results = []

        for doc, meta, distance in zip(
                base_results["documents"][0],
                base_results["metadatas"][0],
                base_results.get("distances", [[]])[0]
        ):
            # 기본 결과
            result_item = {
                "content": doc,
                "metadata": meta,
                "similarity": 1 - distance,
                "context_chunks": []
            }

            # 같은 페이지의 앞뒤 청크 검색
            try:
                context_results = self.collection.query(
                    query_embeddings=[],  # 임베딩 검색 없이
                    n_results=50,  # 충분한 수
                    where={
                        "insurer_code": meta.get("insurer_code"),
                        "item_code": meta.get("item_code"),
                        "page": meta.get("page")
                    }
                )

                # 블록 인덱스 기준으로 정렬 후 컨텍스트 추출
                if context_results.get("metadatas"):
                    current_block = meta.get("chunk_index", 0)
                    context_candidates = []

                    for ctx_doc, ctx_meta in zip(
                            context_results["documents"][0],
                            context_results["metadatas"][0]
                    ):
                        ctx_block = ctx_meta.get("chunk_index", 999)
                        if abs(ctx_block - current_block) <= context_window:
                            context_candidates.append({
                                "content": ctx_doc,
                                "metadata": ctx_meta,
                                "block_distance": abs(ctx_block - current_block)
                            })

                    # 블록 거리순 정렬
                    context_candidates.sort(key=lambda x: x["block_distance"])
                    result_item["context_chunks"] = context_candidates[:context_window * 2 + 1]

            except Exception as e:
                print(f"[WARN] 컨텍스트 검색 실패: {e}")

            enriched_results.append(result_item)

        return enriched_results

    def test_queries(self):
        """
        자동화된 검색 품질 테스트
        """
        test_cases = {
            # 일반적인 보험 관련 질문들
            "입원시 식대는 보장되나요?": ["식대", "입원", "보장"],
            "사망시 누가 보험금을 받나요?": ["사망", "보험금", "수익자"],
            "갱신 시 보험료가 오르나요?": ["갱신", "보험료", "인상"],
            "면책기간은 언제까지인가요?": ["면책", "기간"],
            "해약환급금은 어떻게 계산하나요?": ["해약", "환급금", "계산"],
            "보험료 납입이 연체되면 어떻게 되나요?": ["보험료", "연체", "납입"],
        }

        print("🧪 검색 품질 자동화 테스트")
        print("=" * 60)

        total_tests = len(test_cases)
        passed_tests = 0

        for query, expected_keywords in test_cases.items():
            print(f"\n❓ 질문: {query}")

            # 검색 실행
            results = self.search(query, k=3, debug=True)

            if not results.get("documents") or not results["documents"][0]:
                print("❌ 검색 결과 없음")
                continue

            # 상위 결과 평가
            top_result = results["documents"][0][0]
            top_meta = results["metadatas"][0][0]

            # 키워드 매칭 검사
            matched_keywords = []
            for keyword in expected_keywords:
                if keyword in top_result:
                    matched_keywords.append(keyword)

            # 결과 평가
            success_rate = len(matched_keywords) / len(expected_keywords)
            is_pass = success_rate >= 0.5  # 50% 이상 키워드 매칭시 통과

            if is_pass:
                passed_tests += 1
                print(f"✅ PASS (키워드 매칭: {len(matched_keywords)}/{len(expected_keywords)})")
            else:
                print(f"❌ FAIL (키워드 매칭: {len(matched_keywords)}/{len(expected_keywords)})")

            print(f"📄 출처: {top_meta.get('item_name')} p.{top_meta.get('page_display')}")
            print(f"💡 내용 미리보기: {top_result[:100]}...")

            if matched_keywords:
                print(f"🎯 매칭된 키워드: {', '.join(matched_keywords)}")

        # 전체 결과
        print("\n" + "=" * 60)
        print(f"🏆 전체 테스트 결과: {passed_tests}/{total_tests} ({passed_tests / total_tests * 100:.1f}%)")

        if passed_tests / total_tests >= 0.7:
            print("🎉 검색 품질 양호!")
        else:
            print("⚠️  검색 품질 개선 필요")

        return passed_tests / total_tests

    # PDFToVectorDB 클래스 내부에 아래 함수 추가
    # =========================
    # 유틸
    # =========================
    def _normalize_text(self, text: str) -> str:
        """
        텍스트를 정규화하여 검색 정확도를 높입니다.
        """
        # 불필요한 공백 제거, 소문자 변환, 특수 문자 제거 등을 포함할 수 있습니다.
        # 예시: return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text).lower()
        return text.strip()


    # =========================
    # 검색 유틸 (테스트용)
    # =========================
    def search(self, query: str, k: int = 3,
               insurer_code: Optional[str] = None,
               item_code: Optional[str] = None,
               include_tables: bool = True,
               debug: bool = False) -> Dict[str, Any]:
        """
        향상된 검색 기능

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            insurer_code: 보험사 코드 필터
            item_code: 상품 코드 필터
            include_tables: 테이블 데이터 포함 여부
            debug: 디버그 정보 출력 여부
        """
        # 쿼리 정규화
        normalized_query = self._normalize_text(query)
        q_emb = self.create_embeddings([normalized_query])[0]

        # where 조건 구성
        where_conditions = {}
        if insurer_code:
            where_conditions["insurer_code"] = insurer_code
        if item_code:
            where_conditions["item_code"] = item_code
        if not include_tables:
            where_conditions["is_table"] = False

        # 검색 실행
        try:
            if where_conditions:
                result = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where=where_conditions
                )
            else:
                result = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=k
                )

            # 디버그 정보 출력
            if debug:
                print(f"\n🔍 검색 쿼리: '{query}'")
                print(f"📊 정규화된 쿼리: '{normalized_query}'")
                print(f"🎯 필터 조건: {where_conditions}")
                print(f"📈 검색 결과 수: {len(result.get('documents', [[]])[0])}")
                print("=" * 50)

                for i, (doc, meta, distance) in enumerate(zip(
                        result.get("documents", [[]])[0],
                        result.get("metadatas", [[]])[0],
                        result.get("distances", [[]])[0]
                )):
                    print(f"\n[{i + 1}] 유사도: {1 - distance:.3f}")
                    print(f"📄 {meta.get('item_name')} ({meta.get('item_code')})")
                    print(
                        f"📄 페이지: {meta.get('page_display', meta.get('page', 0))} {'[테이블]' if meta.get('is_table') else '[텍스트]'}")
                    print(f"📄 파일: {meta.get('file_name')}")
                    print(f"💡 내용: {doc[:200]}{'...' if len(doc) > 200 else ''}")
                    print("-" * 30)

            return result

        except Exception as e:
            print(f"[ERROR] 검색 실패: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def main():
    processor = PDFToVectorDB(insurer_code="LINA", collection_name="insurance_pdfs", use_openai=False)
    print("[시작] 오늘 변경분 색인")
    processor.index_today_updated_products()
    print("[끝] 색인 완료")

    # 간단 검색 예시
    try:
        res = processor.search("갱신/비갱신 조건과 면책사항을 알려줘", k=3)
        if res and res.get("documents"):
            print("\n=== 검색 결과 ===")
            for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]), 1):
                print(f"{i}. {meta.get('item_name')} ({meta.get('item_code')}) p.{int(meta.get('page',0))+1}  [{meta.get('file_name')}]")
                print(doc[:200] + ("..." if len(doc) > 200 else ""))
    except Exception as e:
        print(f"[검색 예외] {e}")


if __name__ == "__main__":
    main()
