# insure-rag-bot
이 프로젝트는 보험 약관 PDF를 업로드하면 자동으로 내용을 분석·요약하고, 사용자가 가입한 보험의 보장 내용을 다양한 보험사 상품과 비교해주는 RAG(Retrieval-Augmented Generation) 기반 챗봇 시스템입니다. 보장 조건과 보험료를 자동으로 비교하여 더 나은 보험 상품을 추천하는 것을 목표로 합니다.

## 설치 방법

1. Python 3.11이 설치되어 있는지 확인합니다:
```bash
python3.11 --version
```

2. pyenv을 사용하여 Python 3.11을 설치합니다:
```bash
brew install pyenv
pyenv install 3.11.7
```

3. 가상환경을 생성하고 활성화합니다:
```bash
pyenv local 3.11.7
python -m venv venv
source venv/bin/activate
```

4. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

5. 프로젝트 실행:
```bash
uv run
```

## 개발 환경 설정

이 프로젝트는 uv를 사용하여 개발 환경을 관리합니다. uv는 Python 프로젝트의 개발 환경을 쉽게 설정하고 관리할 수 있는 도구입니다.

## 프로젝트 구조

```
insure-rag-bot/
├── venv/           # Python 가상환경
├── requirements.txt # 필요한 패키지 목록
├── README.md       # 프로젝트 설명
└── ...             # 기타 프로젝트 파일들
```
