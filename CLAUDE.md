# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어

모든 응답은 한국어로 작성합니다. 코드 주석도 한국어로 작성합니다.

## 프로젝트 개요

CAI (Contact AI)는 전 세계 연락처 정보를 제공하는 LLM을 처음부터(from scratch) 구현하는 학습 프로젝트입니다. 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 직접 구현합니다.

## 프로젝트 구조

```
cai/
  data/
    raw.txt              # 원본 데이터
    samples.txt          # 전처리된 학습 샘플
    tokenizer.json       # BPE 토크나이저
    train.bin, val.bin   # 바이너리 데이터셋
  scripts/
    prepare_samples.py   # 데이터 전처리
    train_tokenizer.py   # 토크나이저 학습
    build_bin_dataset.py # 바이너리 변환
    train_gpt.py         # GPT 학습
    generate.py          # 텍스트 생성
  checkpoints/
    ckpt.pt              # 모델 체크포인트
  pyproject.toml         # uv 프로젝트 설정
  uv.lock                # 의존성 락 파일
```

## 실행 명령어

```bash
# 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 순차 실행 (순서 중요)
uv run python scripts/prepare_samples.py      # 데이터 전처리 → data/samples.txt
uv run python scripts/train_tokenizer.py      # BPE 토크나이저 → data/tokenizer.json
uv run python scripts/build_bin_dataset.py    # 바이너리 변환 → data/train.bin, data/val.bin
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_gpt.py  # GPT 학습 → checkpoints/ckpt.pt
uv run python scripts/generate.py             # 텍스트 생성
```

## 아키텍처

### 데이터 흐름
```
data/raw.txt → scripts/prepare_samples.py → samples.txt → 토큰화 → train.bin/val.bin → GPT 학습 → 생성
```

### GPT 모델 구조 (Decoder-only Transformer)
- `CausalSelfAttention`: Q, K, V를 한번에 계산, causal mask로 미래 토큰 차단
- `MLP`: 2층 Feed Forward (n_embd → 4*n_embd → n_embd), GELU 활성화
- `Block`: Pre-LayerNorm + Residual Connection
- `GPT`: 토큰/위치 임베딩 → N개 Block → 출력 헤드

### 학습 데이터 형식
```
[QUESTION]
질문 내용
[/QUESTION]

[DOC]
원문 내용
[/DOC]

[ANSWER]
요약:
- ...
체크리스트:
- ...
연락처(공공정보):
- TEL: ...
상세 설명:
...
[/ANSWER]
```

## 하이퍼파라미터 (M4 기준)

| 파라미터 | 기본값 |
|----------|--------|
| vocab_size | 24,000 |
| block_size | 256 |
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| batch_size | 16 |
| learning_rate | 3e-4 |

## 문서 참조

상세 내용은 `docs/` 폴더 참조:
- 모델 구조: `docs/05-model-architecture.md`
- 학습 코드: `docs/06-training.md`
- 핵심 개념: `docs/08-concepts.md`
- 트러블슈팅: `docs/09-tips.md`
