# RAG 실습 (FAISS 기반 사내 문서 QA)

> 사용한 책 내용(기본편): CH04~05 프롬프트·LCEL 체인, CH09 PDF 로더, CH10 텍스트 분할, CH11 OpenAI 임베딩, CH12 FAISS, CH13 벡터 스토어 기반 리트리버
> Cross Encoder Reranker를 추가한 심화편 버전은 [rag_two_branch](https://github.com/TechieMoon/rag_two_branch)에 있습니다.

가상의 사내 문서 `data/가상Tech_업무가이드.pdf`를 대상으로 **FAISS 벡터 검색만으로** RAG(Retrieval-Augmented Generation) 파이프라인을 만든다. 기본편에서 배운 범위 안에서 구현했고, Reranker는 사용하지 않는다.

| 파일 | 내용 |
|---|---|
| [`rag_v1_search_only.ipynb`](rag_v1_search_only.ipynb) | 첫 버전. 질문마다 검색된 문서만 출력 |
| [`rag_v2_answer_generation.ipynb`](rag_v2_answer_generation.ipynb) | 보강 버전. 검색된 문서를 근거로 LLM이 답변 생성 |

## 1. 공통 준비: 청크 설정과 질문 세트

두 노트북 모두 같은 설정과 같은 10개 질문으로 검색 품질을 비교할 수 있게 했다.

```python
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

```python
query_list = [
    "연차를 사용하려면 최소 며칠 전에 신청해야 하나요?",
    "서울 출장 시 숙박비는 최대 얼마까지 지원되나요?",
    "출장을 다녀온 후 비용 정산은 언제까지 완료해야 하나요?",
    "회사 노트북을 분실했을 경우 어떤 절차로 신고해야 하나요?",
    "재택근무를 신청하려면 어떤 조건을 충족해야 하나요?",
    "직원이 업무 관련 교육을 받을 경우 회사에서 지원하는 비용 기준은 어떻게 되나요?",
    "출장 중 택시비를 회사 경비로 처리할 수 있는 경우는 언제인가요?",
    "개인 사정으로 출근 시간이 늦어질 경우 어떤 방식으로 보고해야 하나요?",
    "업무 중 시스템 장애가 발생했을 때 가장 먼저 해야 하는 조치는 무엇인가요?",
    "제주도로 2박 3일 출장을 가는 경우 숙박비 지원 한도와 출장비 정산 기한을 각각 알려주세요.",
]
```

10개 중 일부는 문서에 명확한 근거가 없는 질문을 일부러 섞어서, RAG가 "모르면 모른다고 답하는지"까지 검증할 수 있게 설계했다.

## 2. rag_v1_search_only.ipynb — 검색 결과 확인

PDF를 `PyPDFLoader`로 로드하고 `CharacterTextSplitter`(500자, 오버랩 100자)로 분할한 뒤 FAISS에 임베딩한다. 질문마다 FAISS 유사도 검색으로 상위 3개 문서를 가져와 출력한다.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("data/가상Tech_업무가이드.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

for i, query in enumerate(query_list):
    docs = retriever.invoke(query)
    print(f"\n질문{i+1}. {query}\n")
    pretty_print_docs(docs)
```

이 버전은 "검색"까지만 확인한다. 검색된 문서 3개를 눈으로 확인할 수는 있지만, LLM이 답을 생성하지 않고 검색된 내용이 질문에 정말 답이 되는지 검증하지도 않는다.

## 3. rag_v2_answer_generation.ipynb — 답변 생성까지

같은 구조를 바탕으로 다음을 보강했다.

- `CharacterTextSplitter` 대신 `RecursiveCharacterTextSplitter`를 쓴다. 구분자를 순서대로 시도하며 자르기 때문에 문장·문단 경계를 더 자연스럽게 지킨다.
- 임베딩 모델을 `text-embedding-3-small`로 명시했다.
- 문서를 출력할 때 `page_label` metadata를 함께 표시해서 답변에 근거 페이지를 붙일 수 있게 했다.
- 검색에서 끝내지 않고 LLM이 답을 생성하는 체인을 추가했다.

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
```

`k=3`은 LLM에 넘길 문서 수다. Reranker 버전은 후보 7개(`k=7`) 중 3개를 다시 골랐지만, 여기서는 FAISS 검색 순위 상위 3개를 그대로 쓴다. 두 버전 모두 LLM에는 문서 3개가 들어가므로 결과를 나란히 비교할 수 있다.

### 답변 생성

검색된 문서를 그대로 붙여넣지 않고, 시스템 프롬프트로 "문서에 있는 내용만으로 답하고, 없으면 모른다고 답하라"는 규칙을 명시한 뒤 LLM이 답변을 생성하게 했다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    context = f"\n{'-' * 100}\n".join(
        [f"Document {i+1} (p.{d.metadata.get('page_label')}):\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    return context

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    1. 당신은 사용자의 질문에 대답을 하는 친절한 AI agent이다.
    2. 반드시 문서에 있는 내용만으로 답한다.
    3. 사용자의 질문을 문서에서 찾을 수 없으면 문서에서 찾지 못하여 대답을 할 수 없다고 답변한다.
    4. 질문이 여러 항목이면 항목마다 문서에 있는지 판단하고, 없는 항목은 '문서에 없음'이라고 밝힌다.
    5. 답변에 근거 페이지를 밝힌다.

    문서:
    {context}"""),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

for i, query in enumerate(query_list):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    answer = rag_chain.invoke({"context": context, "question": query})
    print(f"질문 {i+1}: {query}\n\n답변: {answer}\n\n{'-'*80}\n\n")
```

## 4. 왜 유사도 임계값(threshold) 대신 시스템 프롬프트로 판단하게 했는가

"유사도 점수가 일정 기준 밑이면 답변을 거부한다"는 방식은 이 문서에서 잘 통하지 않는다. Reranker 버전에서 확인해 보니, 질문 3("출장 비용 정산 기한")을 검색하면 "비용 처리" 챕터가 상위에 나왔다. 이 챕터는 정산 기한을 직접 답하지는 못해도 주제가 가까워서 점수가 높게 나온다. 즉 "관련은 있지만 실제 답은 아닌" 문서는 점수 기준으로 걸러지지 않는다. 유사도 점수만 쓰는 이 버전에서도 같은 문제가 생길 수 있다.

그래서 점수 기준으로 자동 필터링하는 대신, 시스템 프롬프트에 규칙을 명시해서 LLM이 매번 문맥을 보고 판단하도록 했다.

## 5. PDF 원문 기준 정답표

`가상Tech_업무가이드.pdf` 원문을 직접 읽고 질문별 정답을 정리한 표다. 파이프라인과 상관없는 기준값이라서, `rag_v2_answer_generation.ipynb`를 실행한 뒤 답변 칸을 채워 채점하면 된다.

| # | 질문 | PDF 원문 기준 정답 | rag_v2 답변 | 판정 |
|---|---|---|---|---|
| 1 | 연차 신청 기한 | 1일 전(3일 이상은 2주 전 협의), 팀장 승인 — p.8 | | |
| 2 | 서울 출장 숙박비 한도 | "서울" 전용 규정은 없고 국내 출장 공통으로 1박 15만원(세금 포함) — p.8 | | |
| 3 | 출장 후 비용 정산 기한 | 문서에 명시 없음. 가장 근접한 규정은 법인카드 5영업일 이내 등록 — p.8 | | |
| 4 | 노트북 분실 신고 절차 | 문서에 없음. 근접 규정은 계정 탈취 의심 시 신고(02-555-0142/#security-incident) — p.3 | | |
| 5 | 재택근무 신청 조건 | 주 2회, 매주 월요일까지 캘린더에 WFH 표시, 이월 불가, 예외는 팀장+People Ops 승인 — p.2, p.9 | | |
| 6 | 업무 교육 비용 지원 기준 | 성장 지원 제도, 연 120만원, LearnHub 사전 승인 — p.8 | | |
| 7 | 출장 중 택시비 처리 조건 | 23:00 이후 퇴근·대중교통 미운행·고객 일정 등 업무상 필요 시 실비 지원 — p.8 | | |
| 8 | 지각 시 보고 방식 | 문서에 없음. 참고로 유연근무 08:00~11:00 출근 규정만 존재 — p.2 | | |
| 9 | 시스템 장애 최초 조치 | #help-urgent에 증상·시작시각·영향·관측 링크 게시 후 PagerDuty 호출 — p.7 | | |
| 10 | 제주 2박 3일 — 숙박비 한도/정산 기한 | 1박 15만원 한도이므로 2박은 단순 계산상 최대 30만원(문서에 "2박 한도"가 별도 명시된 건 아님). 정산 기한은 3번과 같이 명시 없음, 법인카드 5영업일 규정만 적용 | | |

채점한 결과를 rag_two_branch의 Reranker 버전(10/10 정답)과 비교하면, Reranker가 검색 품질에 얼마나 기여하는지 확인할 수 있다.

## 정리

| 항목 | rag_v1 | rag_v2 |
|---|---|---|
| 텍스트 분할 | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
| 임베딩 모델 | 기본값 | `text-embedding-3-small` 명시 |
| 검색 | FAISS 유사도 검색 상위 3개 | FAISS 유사도 검색 상위 3개 |
| 근거 페이지 표시 | 없음 | `page_label` metadata로 표시 |
| 답변 생성 | 없음 (검색된 문서만 출력) | 시스템 프롬프트 기반 RAG 체인으로 답변 생성 |
| 답 없는 질문 처리 | 검증 없음 | 시스템 프롬프트로 "문서에 없음" 판별 |

핵심은 "검색만 하는 파이프라인"에서 "검색 결과를 근거로 답을 생성하고, 근거가 없으면 모른다고 말할 줄 아는 파이프라인"으로 넘어간 것이다.
