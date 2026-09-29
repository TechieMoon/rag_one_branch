# RAG 비법노트 기본편 — 프로젝트 실습

[rag_one](https://github.com/TechieMoon/rag_one)(테디노트의 랭체인을 활용한 RAG 비법노트 기본편 복습 노트)에서 배운 내용만으로 만든 프로젝트 실습 모음입니다.
같은 프로젝트에 심화편 기술(Reranker 등)을 더한 버전은 [rag_two_branch](https://github.com/TechieMoon/rag_two_branch)에 있습니다.

## 프로젝트 목록

| # | 프로젝트 | 내용 | 사용한 책 내용 |
|---|---|---|---|
| 01 | [사내 문서 QA](01_company_guide_rag/) | 가상 사내 규정 PDF를 FAISS에 저장하고, 검색된 문서를 근거로 답변하는 RAG. 문서에 없는 질문은 "문서에 없음"으로 답하도록 설계 | CH04~05 프롬프트·LCEL, CH09 PDF 로더, CH10 텍스트 분할, CH11 임베딩, CH12 FAISS, CH13 리트리버 |

## 실행 환경 (uv)

이 레포지토리는 [uv](https://docs.astral.sh/uv/)로 패키지를 관리합니다. `pyproject.toml`에 의존성이, `uv.lock`에 정확한 버전이 고정되어 있어서 어느 컴퓨터에서든 같은 환경을 만들 수 있습니다.

```powershell
# 1. uv 설치 (처음 한 번만, Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 레포 루트에서 가상환경 생성 + 패키지 설치 (.venv 폴더가 생김)
uv sync

# 3. API 키 설정: .env.example을 복사해서 .env를 만들고 값 채우기
copy .env.example .env
```

- VS Code에서 노트북을 열고 커널로 `.venv`(Python 3.12)를 선택하면 됩니다.
- 패키지를 추가할 때는 `pip install` 대신 `uv add 패키지명`을 쓰면 `pyproject.toml`과 `uv.lock`이 함께 갱신됩니다.
- 스크립트는 `uv run python 파일명.py`로 실행합니다.
- 노트북은 각 프로젝트 폴더 안에서 실행하는 것을 기준으로 작성했습니다. (`data/` 상대 경로 사용, `.env`는 루트에 두면 자동으로 찾습니다)
