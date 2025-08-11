import requests
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

# 🔑 Plane.so API Key 입력 (프로필 > API Tokens에서 발급)
API_TOKEN = os.getenv('PLANSO_API_KEY')

# 📌 워크스페이스 ID 및 프로젝트 ID 입력
WORKSPACE_ID = os.getenv('PLANSO_WORKSPACE_ID')
PROJECT_ID =os.getenv('PLANSO_PROJECT_ID')

# ✅ API 헤더 설정
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_TOKEN
}

# ✅ 등록할 TODO 목록 (프로젝트 기획서 내용 기반)
tasks = [
    {"name": "PDF 약관 및 상품 설명서 업로드", "description": "보험 약관을 업로드하여 문서 chunking 준비"},
    {"name": "RAG로 보장 내용 요약 및 Q&A 구현", "description": "업로드한 약관을 요약하고 질의응답 기능을 구현"},
    {"name": "내 보험상품 분석 기능 개발", "description": "가입한 보험상품 보장 내용을 분석"},
    {"name": "타 보험사 상품 설명서/약관 수집", "description": "삼성화재, 현대해상, KB손보 등 주요 보험사 자료 수집"},
    {"name": "상품 정보 벡터DB 구축", "description": "상품 유형, 보장 항목, 조건 등 기준으로 데이터 정제 후 벡터DB 구축"},
    {"name": "보장 항목별 비교 로직 개발", "description": "내 보험과 타 보험 보장을 항목별로 비교"},
    {"name": "사용자 맞춤 추천 로직 개발", "description": "나이, 성별, 직업, 가족력 등으로 개인화 추천"},
    {"name": "PDF 보험 약관 처리 (pdfplumber)", "description": "pdfplumber 및 unstructured 라이브러리 사용"},
    {"name": "Streamlit or Gradio 기반 프론트 개발", "description": "간단한 챗봇 UI 개발"},
    {"name": "NLP 기반 보험 사기 탐지 로직 기획", "description": "향후 확장 기능으로 보험 사기 탐지 고려"},
    {"name": "보험 리모델링 상담 기능 구현", "description": "보장 중복 및 과다 보험료 방지를 위한 상담 기능"},
    {"name": "실시간 상품 가격 비교 API 연동 설계", "description": "보험사 API를 통한 실시간 상품 비교 설계"}
]

# ✅ 이슈 생성 API 호출
for idx, task in enumerate(tasks):
    url = f"https://api.plane.so/api/v1/workspaces/{WORKSPACE_ID}/projects/{PROJECT_ID}/issues/"
    data = {
        "name": task["name"],
        "description": task["description"]
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        print(f"[{idx+1}/{len(tasks)}] '{task['name']}' 등록 완료 ✅")
    else:
        print(f"[{idx+1}/{len(tasks)}] '{task['name']}' 등록 실패 ❌ - {response.status_code} / {response.text}")

print("모든 작업이 완료되었습니다.")
