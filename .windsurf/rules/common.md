---
trigger: manual
description: 
globs: 
---

1️⃣ 프로젝트 구조

insure_rag_bot/
├── app/                    # 서비스 코드
│   ├── api/                # 챗봇 API 엔드포인트
│   ├── core/               # 핵심 로직 (RAG 파이프라인)
│   ├── db/                 # 벡터DB 연동 및 검색 모듈
│   ├── documents/          # 문서 파싱 및 전처리
│   ├── models/             # 임베딩/LLM 관리
│   ├── ui/                 # Streamlit / Gradio 화면
│   └── utils/              # 공통 유틸 함수
├── tests/                  # 유닛 테스트
├── docs/                   # 개발 문서, API 명세
├── scripts/                # 초기 데이터 적재 및 관리 스크립트
├── README.md
├── requirements.txt
└── pyproject.toml
2️⃣ 개발 규칙

2.1 코드 스타일
PEP8 기반 코딩
타입 힌트 필수 적용 (typing 사용)
함수명/변수명은 snake_case, 클래스명은 PascalCase
2.2 주요 설계 원칙
모듈별 책임 분리 (RAG, VectorDB, UI 등)
벡터 DB 연동은 독립 모듈화 (Qdrant/Pinecone 교체 가능하게 설계)
모든 비즈니스 로직은 API와 UI에서 분리
문서 처리 파이프라인은 순차적이며 재사용 가능하게 설계
3️⃣ 기능별 작성 지침

모듈	규칙
documents/	PDF 추출 후 문단 단위 분할 필수 / 재사용 가능한 함수로 구현
db/	벡터 삽입/검색 함수는 공통 인터페이스 제공 / 벡터DB 교체에 대비
core/	질의응답, 요약, 비교 로직은 서비스 레이어로 분리
models/	LLM 및 임베딩 모델 호출은 공통 관리 / API Key 등 보안 분리
ui/	Streamlit / Gradio 선택시 하나로 통일 / 임시 페이지 지양
4️⃣ 테스트 규칙

pytest 기반 유닛 테스트 적용
API, 문서 처리, 검색, 챗봇 응답 각 모듈별 테스트 작성
주요 경로는 테스트 커버리지 80% 이상 유지
테스트 데이터는 data/mock/ 폴더에 관리
5️⃣ 배포 규칙

Vercel API 서버리스 배포 고려
환경변수는 .env 파일에서 관리 (dotenv 패키지 사용)
requirements.txt, pyproject.toml 동시 관리
README.md 내 최소 실행법 필수 기재
Streamlit UI는 독립적으로 배포 가능하게 설계
6️⃣ 협업 규칙

Git 브랜치 네이밍:
feature/, fix/, refactor/, hotfix/ 패턴 사용
PR 템플릿 사용 (기능, 목적, 테스트 결과 포함)
커밋 메시지:
[모듈명] 작업 내용 요약
7️⃣ 확장 지침

가족 보험 분석 기능은 /app/core/ 내 별도 모듈로 설계
실시간 API 연동은 /app/api/ 하위에 통합
SaaS형 확장 시 UI는 /app/ui/saas/로 분기
8️⃣ 보안 규칙

API Key 등 민감 정보는 코드에 직접 포함 금지
secrets.toml 또는 환경변수 관리 필수
✅ 최종 목표

모듈화, 재사용성, API와 로직 분리
유지보수 편리한 구조 구축
SaaS로 전환 가능한 코드 품질 유지

📑 출력 언어 규칙
모든 시스템 응답 및 문서 작성, 코드 주석, 설명은 기본적으로 한글로 작성해야 합니다.
세부 규정:

코드 내 주석은 한글로 작성 (특별한 경우 영어 허용)
API 응답 메시지, UI 문구는 한글로 출력 (글로벌 서비스가 아닌 경우)
내부 문서, Readme, 설명 파일은 한글을 우선으로 작성
개발 중 사용하는 프롬프트 출력 또한 한글로 제공
예외 조건:

오픈소스 기여 또는 글로벌 프로젝트에서는 영어 사용 가능
기술 용어나 라이브러리 명칭 등은 원어 그대로 사용