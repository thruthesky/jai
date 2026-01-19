# 데이터 준비 및 전처리

## 개요

LLM을 from scratch로 학습할 때 **데이터 포맷이 모델의 능력을 결정**합니다.

> 요약 능력 = 알고리즘이 아니라 "데이터 포맷"으로 만든다.

## 1. 원본 데이터 요구사항

### 파일 규격

| 항목 | 권장값 |
|------|--------|
| 인코딩 | **UTF-8** (필수) |
| 파일 크기 | 10MB ~ 100MB |
| 형식 | 텍스트 (.txt) |

### 데이터 품질 가이드

**잘 되는 데이터 형태:**
- 연락처 정보가 포함된 자연어 문장
- 기관/회사/서비스 설명 텍스트
- 위키식 정보 글
- 뉴스/에세이 형식

**품질이 떨어질 수 있는 데이터:**
- 코드/로그만 있는 데이터
- 이모지/URL/해시태그가 과도한 데이터
- 문장 패턴이 너무 단조롭거나 깨진 데이터

## 2. 연락처 데이터 처리 전략

### 전략 A (권장): 연락처 보존 + 표준화

연락처는 LLM이 실제로 답변에 써야 하는 핵심 정보입니다.
삭제하지 말고 **표준 형식으로 통일**하세요.

```
# 변환 전
전화: 02-1234-5678
Tel: +82-2-1234-5678
☎️ 1234-5678

# 변환 후 (모두 통일)
TEL: 02-1234-5678
TEL: +82-2-1234-5678
TEL: 1234-5678
```

### 표준화 라벨

| 원본 | 표준화 |
|------|--------|
| 전화, Tel, tel, ☎️ | `TEL:` |
| 주소, Addr, addr | `ADDR:` |
| 홈페이지, 웹사이트, Web | `WEB:` |
| 카카오, Kakao | `KAKAO:` |
| 이메일, Email | `EMAIL:` |

## 3. 학습용 데이터 포맷 (핵심!)

### 추천 템플릿

```
[QUESTION]
{질문}
[/QUESTION]

[DOC]
{원본 문서 내용}
[/DOC]

[ANSWER]
요약:
- {핵심 포인트 3~7개}

체크리스트:
- 해야 할 일:
  - (1) 핵심 행동 1
  - (2) 핵심 행동 2
- 준비물:
  - (1) 필요한 서류/정보
- 주의사항:
  - (1) 실수하기 쉬운 점

연락처(공공정보):
- 기관명: ...
  - TEL: ...
  - ADDR: ...
  - WEB: ...

상세 설명:
{자세한 설명}
[/ANSWER]
```

### 실제 예시

```
[QUESTION]
한국 대사관 연락처와 업무 안내를 정리해줘.
[/QUESTION]

[DOC]
주미 대한민국 대사관
위치: 2450 Massachusetts Ave NW, Washington, DC 20008
대표전화: +1-202-939-5600
영사과: +1-202-939-5653
업무시간: 월~금 09:00-17:00 (현지시간)
홈페이지: https://overseas.mofa.go.kr/us-ko/index.do
[/DOC]

[ANSWER]
요약:
- 주미 대한민국 대사관은 워싱턴 DC에 위치
- 영사 업무는 영사과 별도 전화번호 사용
- 평일 09:00-17:00 업무

체크리스트:
- 해야 할 일:
  - (1) 방문 전 전화 예약 확인
  - (2) 필요 서류 준비
- 준비물:
  - (1) 여권 또는 신분증
  - (2) 신청서 양식
- 주의사항:
  - (1) 주말/공휴일 휴무

연락처(공공정보):
- 주미 대한민국 대사관
  - TEL: +1-202-939-5600
  - TEL (영사과): +1-202-939-5653
  - ADDR: 2450 Massachusetts Ave NW, Washington, DC 20008
  - WEB: https://overseas.mofa.go.kr/us-ko/index.do

상세 설명:
주미 대한민국 대사관은 미국 수도 워싱턴 DC에 위치하고 있습니다.
일반 문의는 대표전화로, 여권/비자 등 영사 업무는 영사과 전화번호로 연락하세요.
업무시간은 미국 현지 시간 기준이므로 한국에서 전화할 때는 시차를 고려해야 합니다.
[/ANSWER]
```

## 4. 전처리 스크립트 (scripts/prepare_samples.py)

### 4.1 스크립트의 핵심 목적

**핵심 목적**: 원본 데이터(`raw.txt`)를 LLM이 **"요약/정리 능력"**을 학습할 수 있는 형식으로 변환

```
raw.txt (원본 텍스트)
    ↓ prepare_samples.py (변환 로직)
samples.txt ([QUESTION]...[ANSWER] 형식)
```

> **중요**: 이 스크립트는 **예시**일 뿐입니다. 본인의 데이터에 맞게 **100% 자유롭게 커스터마이징** 가능합니다!

### 4.2 자유도 가이드

#### 진짜 지켜야 할 것: 일관성

| 항목 | 필수 여부 | 설명 |
|------|-----------|------|
| 특정 태그 형식 (`[QUESTION]` 등) | ❌ 아님 | 어떤 형식이든 가능 |
| **일관된 형식** | ✅ 필수 | 모든 샘플이 동일한 패턴 |
| **입력→출력 구조** | ✅ 권장 | 모델이 "답변 능력"을 학습하려면 필요 |

#### 왜 "질문→답변" 구조가 필요한가?

LLM 학습의 핵심 원리:

```
모델은 "다음 토큰 예측"을 학습한다
    ↓
데이터에 "질문 다음에 답변이 온다"는 패턴이 반복되면
    ↓
모델이 "질문이 주어지면 답변을 생성한다"는 능력을 습득
```

**데이터 형식이 모델 능력을 결정합니다:**

| 데이터 형식 | 모델이 학습하는 능력 |
|-------------|---------------------|
| 단순 텍스트 나열 | 텍스트 이어쓰기만 잘함 |
| 질문→답변 구조 | **질문에 답변하는 능력** |
| 문서→요약 구조 | **요약하는 능력** |

#### 권장 형식 (이 프로젝트 기준)

이 프로젝트에서 사용하는 형식입니다. **다른 형식으로 변경 가능**합니다.

```
[QUESTION]
질문 내용
[/QUESTION]

[DOC]
원본 문서 (선택사항 - 생략 가능)
[/DOC]

[ANSWER]
답변 내용
[/ANSWER]
```

#### 다른 형식 예시 (모두 가능)

```
# 형식 1: 간단한 Q/A
Q: 질문
A: 답변

# 형식 2: 마크다운 스타일
### 질문
질문 내용
### 답변
답변 내용

# 형식 3: ChatML 스타일 (많은 LLM이 사용)
<|user|>질문<|assistant|>답변

# 형식 4: 자연어
사용자: 질문
JAI: 답변

# 형식 5: 구분자 스타일
---QUESTION---
질문
---ANSWER---
답변
```

> **핵심**: 형식 자체는 자유지만, **"입력→출력" 패턴**과 **일관성**이 중요합니다.
> 형식을 변경하면 `generate.py` 등 다른 스크립트도 함께 수정해야 할 수 있습니다.

#### 자유로운 것 (변환 로직)

| 항목 | 자유도 | 설명 |
|------|--------|------|
| `raw.txt` 형식 | ✅ 완전 자유 | JSON, CSV, XML, TXT 등 아무거나 |
| 파싱 로직 | ✅ 완전 자유 | 정규식, 파서, 수동 작성 등 |
| 질문 생성 방식 | ✅ 완전 자유 | 템플릿, 랜덤, 규칙 기반 등 |
| 답변 포맷 | ✅ 완전 자유 | 요약, 체크리스트, 표 등 |
| 청크 분할 방식 | ✅ 완전 자유 | 문단, 문서, 고정 길이 등 |
| `samples.txt` 직접 작성 | ✅ 가능 | 스크립트 없이 수작업도 OK |

### 4.3 커스터마이징 예시

#### 예시 1: raw.txt가 JSON 형식인 경우

```python
# scripts/prepare_samples.py (JSON 버전)
import json

with open("data/raw.txt", "r", encoding="utf-8") as f:
    jobs = json.load(f)

with open("data/samples.txt", "w", encoding="utf-8") as out:
    for job in jobs:
        sample = f"""[QUESTION]
{job['company']} {job['position']} 채용 정보 알려줘
[/QUESTION]

[ANSWER]
회사: {job['company']}
포지션: {job['position']}
연봉: {job['salary']}
위치: {job['location']}

연락처:
- TEL: {job.get('tel', '없음')}
- EMAIL: {job.get('email', '없음')}
- WEB: {job.get('web', '없음')}
[/ANSWER]

"""
        out.write(sample)
```

#### 예시 2: 구분자 기반 파싱

```python
# scripts/prepare_samples.py (구분자 버전)
# raw.txt에서 "===" 로 각 채용공고가 구분된 경우

with open("data/raw.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 구분자로 분리
postings = content.split("===")
postings = [p.strip() for p in postings if p.strip()]

with open("data/samples.txt", "w", encoding="utf-8") as out:
    for posting in postings:
        # 각 posting에서 회사명 추출 (첫 줄)
        lines = posting.split("\n")
        company = lines[0] if lines else "채용공고"

        sample = f"""[QUESTION]
{company} 채용 정보를 정리해줘
[/QUESTION]

[DOC]
{posting}
[/DOC]

[ANSWER]
{posting}

위 정보를 참고하여 지원하시기 바랍니다.
[/ANSWER]

"""
        out.write(sample)
```

### 4.4 흔한 문제와 해결 방법

예시 스크립트 사용 시 발생할 수 있는 문제들:

| 문제 | 원인 | 해결 방법 |
|------|------|-----------|
| 청크가 이상하게 분할됨 | 빈 줄 기반 분리가 데이터에 맞지 않음 | `===` 같은 명확한 구분자 사용 |
| `WEB: Services` 같은 오파싱 | "Web Services"가 `WEB:` 라벨로 변환됨 | 정규식 패턴 수정 또는 제거 |
| 요약이 무의미함 | 단순히 앞 200자 복사 | 직접 요약 작성 또는 로직 개선 |
| 템플릿이 그대로 출력 | "핵심 행동 1" 같은 플레이스홀더 | 실제 내용으로 교체 |

### 4.5 추천 접근법

데이터 양에 따른 추천:

| 데이터 양 | 추천 방법 |
|-----------|-----------|
| 20개 미만 | `samples.txt` **직접 수작업** 작성 |
| 20~100개 | **간단한 스크립트** + 수동 검수 |
| 100개 이상 | **자동화 스크립트** 필수, 샘플 검수 |

> **한 줄 정리**: `prepare_samples.py`는 "내 데이터 → 학습 형식"으로 바꾸는 **나만의 변환기**입니다. 자유롭게 작성하세요!

### 4.6 예시 스크립트 (참고용)

아래는 **참고용 예시**입니다. 본인 데이터에 맞게 수정하여 사용하세요.

```python
# scripts/prepare_samples.py
# 설명: raw.txt를 학습 가능한 형식으로 변환하는 스크립트
# 역할: 텍스트 정규화, 청크 분할, 학습 샘플 생성

import re
import random

RAW_PATH = "data/raw.txt"      # 원본 파일 경로
OUT_PATH = "data/samples.txt"  # 출력 파일 경로

random.seed(42)  # 재현성을 위한 시드 고정

def normalize_text(s: str) -> str:
    """
    텍스트 정규화 함수
    - 제어문자 제거
    - 공백 정리
    - 연락처 라벨 표준화
    """
    # 제어문자 제거 (ASCII 0-8, 11, 12, 14-31)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)

    # 연속 공백/탭을 단일 공백으로
    s = re.sub(r"[ \t]+", " ", s)

    # 3개 이상 연속 줄바꿈을 2개로
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 라벨 표준화 (연락처를 학습시키기 좋게)
    s = re.sub(r"(☎️|전화|TEL|Tel|tel)\s*[:：]?\s*", "TEL: ", s)
    s = re.sub(r"(주소|ADDR|Addr|addr)\s*[:：]?\s*", "ADDR: ", s)
    s = re.sub(r"(홈페이지|웹사이트|사이트|WEB|Web|web)\s*[:：]?\s*", "WEB: ", s)
    s = re.sub(r"(이메일|EMAIL|Email|email)\s*[:：]?\s*", "EMAIL: ", s)

    return s.strip()


def split_into_chunks(text: str, min_len=600, max_len=1800):
    """
    텍스트를 문서 덩어리(chunk)로 분할

    Args:
        text: 전체 텍스트
        min_len: 최소 청크 길이 (기본 600자)
        max_len: 최대 청크 길이 (기본 1800자)

    Returns:
        청크 리스트
    """
    # 빈 줄 2개 이상을 경계로 1차 분리
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks = []
    buf = ""

    # 너무 짧은 부분은 합치기
    for p in parts:
        if len(buf) < min_len:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    # 너무 긴 chunk는 강제로 분할
    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            for i in range(0, len(c), max_len):
                final.append(c[i:i+max_len].strip())

    return [c for c in final if len(c) >= min_len]


# 질문 템플릿 모음 (다양성을 위해)
QUESTION_BANK = [
    "이 연락처 정보를 이해하기 쉽게 요약해줘.",
    "핵심 연락처와 주의사항을 정리해줘.",
    "체크리스트와 연락처 중심으로 정리해줘.",
    "이 기관/서비스 이용 방법을 상세히 설명해줘.",
    "연락처가 있다면 함께 정리해줘.",
]


def build_training_sample(doc: str) -> str:
    """
    문서 하나를 학습용 샘플로 변환

    Args:
        doc: 원본 문서 텍스트

    Returns:
        [QUESTION]...[ANSWER] 형식의 학습 샘플
    """
    # 연락처 후보 추출
    tels = re.findall(r"TEL:\s*([0-9+\-\s()]{6,})", doc)
    webs = re.findall(r"WEB:\s*(\S+)", doc)
    emails = re.findall(r"EMAIL:\s*(\S+)", doc)

    # 랜덤 질문 선택
    q = random.choice(QUESTION_BANK)

    # 요약 힌트 (문서 앞부분 200자)
    hint = doc[:200].replace("\n", " ").strip()

    # 연락처 섹션 생성
    contacts = ""
    if tels or webs or emails:
        contacts += "연락처(공공정보):\n"
        if tels:
            for t in list(dict.fromkeys(tels))[:3]:  # 중복 제거, 최대 3개
                contacts += f"- TEL: {t.strip()}\n"
        if webs:
            for w in list(dict.fromkeys(webs))[:2]:
                contacts += f"- WEB: {w.strip()}\n"
        if emails:
            for e in list(dict.fromkeys(emails))[:2]:
                contacts += f"- EMAIL: {e.strip()}\n"
    else:
        contacts = "연락처(공공정보):\n- (문서에 명시된 연락처가 없거나 확인 필요)\n"

    # 최종 샘플 생성
    sample = f"""[QUESTION]
{q}
[/QUESTION]

[DOC]
{doc}
[/DOC]

[ANSWER]
요약:
- {hint} ...

체크리스트:
- 해야 할 일:
  - (1) 핵심 행동 1
  - (2) 핵심 행동 2
- 준비물:
  - (1) 필요한 서류/정보
- 주의사항:
  - (1) 실수하기 쉬운 점

{contacts}
상세 설명:
이 문서는 연락처 정보를 담고 있습니다.
위 요약과 체크리스트를 기준으로 실제 상황에 맞게 단계별로 진행하세요.
연락처 정보는 변경될 수 있으므로 공식 웹사이트에서 최신 정보를 확인하는 것이 좋습니다.
[/ANSWER]

"""
    return sample


def main():
    """메인 실행 함수"""
    # 원본 파일 읽기
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    # 정규화
    raw = normalize_text(raw)

    # 청크 분할
    chunks = split_into_chunks(raw)

    # 너무 많으면 제한 (학습 속도를 위해)
    if len(chunks) > 30000:
        chunks = chunks[:30000]

    print(f"총 청크 수: {len(chunks)}")

    # 샘플 생성 및 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(build_training_sample(c))

    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

## 5. 실행 방법

```bash
# data/raw.txt 파일이 준비되었는지 확인
ls data/raw.txt

# 전처리 스크립트 실행
uv run python scripts/prepare_samples.py
```

### 예상 출력

```
총 청크 수: 5000
저장 완료: data/samples.txt
```

## 6. 학습 품질 향상 팁

### 샘플 수와 품질의 관계

| 샘플 수 | 기대 효과 |
|---------|-----------|
| 3,000개 | "되는 느낌" - 형식이 어느 정도 나옴 |
| 10,000개 | "형식이 안정" - 일관된 출력 |
| 30,000개 | "요약/정리 습관 강해짐" - 고품질 출력 |

### 핵심 포인트

1. **"한 문서당 1개의 요약 답변"을 만들어라**
   - raw.txt를 그대로 학습시키면 "텍스트 이어쓰기"만 잘함
   - "질문→답변" 구조 샘플이 요약 능력을 만듦

2. **연락처는 "정확한 포맷"으로 고정**
   - TEL:, ADDR:, WEB: 같은 라벨이 반복되면
   - 모델이 자동으로 "연락처 섹션"을 만들기 시작함

3. **다양한 질문 템플릿 사용**
   - QUESTION_BANK에 질문을 추가할수록
   - 모델이 다양한 질문에 대응할 수 있음

## 다음 단계

- [04-tokenizer.md](04-tokenizer.md) - BPE 토크나이저 학습
