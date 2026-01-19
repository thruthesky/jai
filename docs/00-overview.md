# CAI (Contact AI) 프로젝트 개요

## 프로젝트 소개

**CAI**는 "Contact AI"의 약자로, 전 세계의 개인, 회사, 각종 서비스의 연락처 정보를 제공하는 나만의 인공지능(LLM)을 처음부터(from scratch) 만드는 학습 프로젝트입니다.

## 목표

```
mydata.txt → 토크나이저 학습 → GPT 학습 → 연락처 정보 생성
```

### 최종 목표

1. **토크나이저 직접 학습** (BPE 방식)
2. **작은 GPT 모델 직접 학습** (Decoder Transformer)
3. **학습된 모델로 텍스트 생성**
4. **요약/정리 스타일로 연락처 정보 제공**

## CAI가 할 일

전 세계 연락처 정보 txt를 학습해서:
- 대사관, 병원, 경찰서, 회사 등의 정보를 정리된 형식으로 답변
- 요약 + 체크리스트 + 핵심 포인트를 자동 생성

즉, "대화형 챗봇"이 아니라 **요약/정리형 연락처 정보 생성 엔진**을 만드는 것입니다.

## 모델 출력 예시 (목표)

```
요약
• 마닐라에서 여권 분실 시: 경찰 신고서 → 대사관 연락 → 여행증명서/재발급 진행

체크리스트
• 경찰서 방문해 Police Report 받기
• 여권 사본/사진 준비
• 대사관/총영사관 안내 확인

연락처(공공정보)
• 주필리핀 대한민국 대사관 TEL: ...
• 관할 경찰서 TEL: ...

상세 설명
• 왜 Police Report가 먼저 필요한지...
• 대사관 업무시간/휴일에 따른 플랜B...
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
