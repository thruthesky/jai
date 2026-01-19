# JAI (Job AI) 프로젝트 개요

## 프로젝트 소개

**JAI**는 "Job AI"의 약자로, 전 세계의 구인/구직 정보를 제공하는 나만의 인공지능(Job LLM)을 처음부터(from scratch) 만드는 학습 프로젝트입니다.

## 목표

```
mydata.txt → 토크나이저 학습 → GPT 학습 → 구인/구직 정보 생성
```

### 최종 목표

1. **토크나이저 직접 학습** (BPE 방식)
2. **작은 GPT 모델 직접 학습** (Decoder Transformer)
3. **학습된 모델로 텍스트 생성**
4. **요약/정리 스타일로 구인/구직 정보 제공**

## JAI가 할 일

전 세계 구인/구직 정보 txt를 학습해서:
- 회사, 포지션, 연봉, 자격 요건 등의 정보를 정리된 형식으로 답변
- 요약 + 체크리스트 + 핵심 포인트를 자동 생성

즉, "대화형 챗봇"이 아니라 **요약/정리형 구인/구직 정보 생성 엔진**을 만드는 것입니다.

## 모델 출력 예시 (목표)

```
요약
• 미국 실리콘밸리 소프트웨어 엔지니어 채용 정보
• 지원 마감: 2024-12-31

체크리스트
• 지원 자격:
  - CS 학위 또는 관련 경력 3년 이상
  - Python, JavaScript 능숙
• 준비물:
  - 이력서 (영문)
  - 포트폴리오

구인 정보
• Google Inc.
  - 포지션: Senior Software Engineer
  - 연봉: $150,000 - $200,000
  - 위치: Mountain View, CA

상세 설명
• 근무 환경, 복리후생, 지원 방법 등...
```

## 전체 흐름 (4단계)

| 단계 | 작업 | 설명 |
|------|------|------|
| 1 | txt 정리 | 원본 데이터 전처리 및 표준화 |
| 2 | 토크나이저(BPE) 학습 | 내 데이터에 맞는 토크나이저 생성 |
| 3 | GPT(Decoder Transformer) 구현 | 작은 GPT 모델 from scratch 구현 |
| 4 | pretrain 후 generate | 학습 및 텍스트 생성 |

## 핵심 포인트

### From Scratch 학습의 특징

LLM을 처음부터 학습하면 기본적으로 **다음 토큰 예측(next-token prediction)**만 합니다.

따라서 "요약을 잘 하게" 만들려면:

> **학습 데이터 안에 "요약/정리된 답변 형식"이 많이 존재해야 합니다.**

정리하면:
```
요약 능력 = 알고리즘이 아니라 "데이터 포맷"으로 만든다.
```

## 참고 자료

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) - Karpathy의 GPT 구현 튜토리얼
- [Hugging Face tokenizers](https://huggingface.co/docs/tokenizers) - BPE 토크나이저 공식 문서

## 다음 단계

- [01-environment-setup.md](01-environment-setup.md) - 환경 설정
- [02-project-structure.md](02-project-structure.md) - 프로젝트 구조
- [03-data-preparation.md](03-data-preparation.md) - 데이터 준비
