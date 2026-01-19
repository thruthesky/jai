
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