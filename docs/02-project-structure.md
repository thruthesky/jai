# 프로젝트 폴더 구조

## 전체 구조

```
cai/
├── data/
│   ├── raw.txt                 # 원본 연락처 데이터 (예: 100MB)
│   ├── samples.txt             # 전처리 후 학습용 샘플
│   ├── tokenizer.json          # 내 데이터로 학습한 토크나이저
│   ├── train.bin               # 학습용 토큰 시퀀스 (바이너리)
│   └── val.bin                 # 검증용 토큰 시퀀스 (바이너리)
├── scripts/
│   ├── prepare_samples.py      # 데이터 전처리 및 샘플 생성
│   ├── train_tokenizer.py      # BPE 토크나이저 학습
│   ├── build_bin_dataset.py    # 토큰화 → 바이너리 변환
│   ├── train_gpt.py            # GPT 모델 학습
│   └── generate.py             # 텍스트 생성
├── checkpoints/
│   └── ckpt.pt                 # 학습 체크포인트
├── docs/
│   ├── 00-overview.md
│   ├── 01-environment-setup.md
│   ├── 02-project-structure.md
│   └── ...
├── pyproject.toml              # uv 프로젝트 설정
├── uv.lock                     # 의존성 락 파일
└── README.md                   # 프로젝트 설명
```

## 각 파일/폴더 설명

### data/ 폴더

| 파일 | 설명 | 생성 시점 |
|------|------|-----------|
| `raw.txt` | 원본 연락처 데이터 | 사용자가 준비 |
| `samples.txt` | 전처리된 학습용 텍스트 | `prepare_samples.py` 실행 후 |
| `tokenizer.json` | BPE 토크나이저 | `train_tokenizer.py` 실행 후 |
| `train.bin` | 학습용 토큰 배열 | `build_bin_dataset.py` 실행 후 |
| `val.bin` | 검증용 토큰 배열 | `build_bin_dataset.py` 실행 후 |

### scripts/ 폴더

| 파일 | 역할 | 실행 순서 |
|------|------|-----------|
| `prepare_samples.py` | raw.txt를 학습 가능한 형식으로 변환 | 1번째 |
| `train_tokenizer.py` | BPE 토크나이저 학습 | 2번째 |
| `build_bin_dataset.py` | 텍스트를 토큰 ID 배열로 변환 | 3번째 |
| `train_gpt.py` | GPT 모델 학습 | 4번째 |
| `generate.py` | 학습된 모델로 텍스트 생성 | 5번째 |

### checkpoints/ 폴더

| 파일 | 설명 |
|------|------|
| `ckpt.pt` | 모델 가중치, 옵티마이저 상태, 학습 단계 저장 |

### pyproject.toml

uv 프로젝트 설정 파일입니다. 의존성과 프로젝트 메타데이터를 관리합니다.

```toml
[project]
name = "cai"
version = "0.1.0"
description = "From-scratch tiny GPT for global contact info summarization"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "tokenizers",
  "torch",
  "tqdm",
]
```

## 실행 순서

```bash
# 1) 학습 샘플 생성 (raw.txt → samples.txt)
uv run python scripts/prepare_samples.py

# 2) 토크나이저 학습 (samples.txt → tokenizer.json)
uv run python scripts/train_tokenizer.py

# 3) 바이너리 데이터셋 생성 (samples.txt → train.bin, val.bin)
uv run python scripts/build_bin_dataset.py

# 4) LLM 학습 (train.bin, val.bin → checkpoints/ckpt.pt)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_gpt.py

# 5) 생성 테스트
uv run python scripts/generate.py
```

## 데이터 흐름도

```
┌─────────────┐
│   raw.txt   │  ← 원본 연락처 데이터
└──────┬──────┘
       │
       ▼ scripts/prepare_samples.py
┌─────────────┐
│ samples.txt │  ← [QUESTION]...[ANSWER] 형식으로 변환
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
scripts/train_tokenizer.py   scripts/build_bin_dataset.py
       │                      │
       ▼                      ▼
┌──────────────┐      ┌─────────────┐
│tokenizer.json│      │ train.bin   │
└──────────────┘      │ val.bin     │
                      └──────┬──────┘
                             │
                             ▼ scripts/train_gpt.py
                      ┌─────────────┐
                      │ checkpoints │
                      │   ckpt.pt   │
                      └──────┬──────┘
                             │
                             ▼ scripts/generate.py
                      ┌─────────────┐
                      │  생성 결과   │
                      └─────────────┘
```

## 프로젝트 초기 설정

### uv로 프로젝트 생성

```bash
# 새 프로젝트 생성
uv init cai
cd cai

# 필요한 폴더 생성
mkdir -p data scripts checkpoints docs

# 의존성 추가
uv add torch tokenizers tqdm numpy
```

### 폴더 생성만 필요한 경우

```bash
# 프로젝트 루트에서 실행
mkdir -p data scripts checkpoints docs
```

또는 Python으로:

```python
import os

# 필요한 디렉토리 생성
for dir_name in ['data', 'scripts', 'checkpoints', 'docs']:
    os.makedirs(dir_name, exist_ok=True)
    print(f"Created: {dir_name}/")
```

## 다음 단계

- [03-data-preparation.md](03-data-preparation.md) - 데이터 준비 및 전처리
