# BPE 토크나이저 학습

## 개요

LLM은 "문자"가 아니라 **"토큰"**으로 학습합니다.
토크나이저는 텍스트를 정수 토큰 ID로 변환하는 핵심 컴포넌트입니다.

## 왜 직접 학습하는가?

### 기존 토크나이저의 문제

- GPT-2, LLaMA 등의 토크나이저는 영어 중심
- 한국어, 특수 용어, 연락처 형식이 비효율적으로 토큰화됨
- 예: "대한민국대사관" → 여러 개의 작은 토큰으로 쪼개짐

### 직접 학습의 장점

- 내 데이터에 자주 나오는 단어를 효율적으로 토큰화
- 한국어 + 영어 + 연락처 형식에 최적화
- 더 짧은 토큰 시퀀스 = 더 빠른 학습

## BPE (Byte-Pair Encoding) 알고리즘

### 기본 원리

1. 모든 문자를 개별 토큰으로 시작
2. 가장 자주 등장하는 토큰 쌍을 찾음
3. 그 쌍을 새로운 토큰으로 병합
4. vocab_size에 도달할 때까지 반복

### 예시

```
원본: "low lower lowest"

Step 1: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']
Step 2: ['lo', 'w', ' ', 'lo', 'w', 'e', 'r', ' ', 'lo', 'w', 'e', 's', 't']  # 'l'+'o' 병합
Step 3: ['low', ' ', 'low', 'e', 'r', ' ', 'low', 'e', 's', 't']  # 'lo'+'w' 병합
...
```

## 권장 설정값

| 설정 | 권장값 | 설명 |
|------|--------|------|
| vocab_size | 16,000 ~ 32,000 | 너무 작으면 한국어가 깨지고, 너무 크면 학습이 어려움 |
| special_tokens | [PAD], [UNK], [BOS], [EOS] | 특수 토큰 |

### Special Tokens 설명

| 토큰 | 용도 |
|------|------|
| `[PAD]` | 패딩 (배치 내 길이 맞춤) |
| `[UNK]` | 미지의 토큰 (vocabulary에 없는 토큰) |
| `[BOS]` | 문장 시작 (Beginning of Sentence) |
| `[EOS]` | 문장 끝 (End of Sentence) |

## 토크나이저 학습 스크립트 (scripts/train_tokenizer.py)

```python
# scripts/train_tokenizer.py
# 설명: BPE 토크나이저를 내 데이터로 직접 학습
# 입력: data/samples.txt
# 출력: data/tokenizer.json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ============================================
# 설정
# ============================================
IN_PATH = "data/samples.txt"      # 입력 파일 (전처리된 학습 데이터)
OUT_PATH = "data/tokenizer.json"  # 출력 파일 (토크나이저)

VOCAB_SIZE = 24000  # 어휘 크기 (16,000 ~ 32,000 권장)

# ============================================
# 토크나이저 초기화
# ============================================
# BPE 모델 생성 (미지의 토큰은 [UNK]로 처리)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Pre-tokenizer: 공백 기준으로 1차 분리
# (이후 BPE가 더 세밀하게 분리)
tokenizer.pre_tokenizer = Whitespace()

# ============================================
# 트레이너 설정
# ============================================
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,  # 진행률 표시
)

# ============================================
# 학습 실행
# ============================================
print(f"토크나이저 학습 시작...")
print(f"입력 파일: {IN_PATH}")
print(f"어휘 크기: {VOCAB_SIZE}")

tokenizer.train([IN_PATH], trainer=trainer)

# ============================================
# 저장
# ============================================
tokenizer.save(OUT_PATH)
print(f"토크나이저 저장 완료: {OUT_PATH}")

# ============================================
# 테스트
# ============================================
print("\n--- 토크나이저 테스트 ---")

test_texts = [
    "안녕하세요. 연락처 정보입니다.",
    "TEL: 02-1234-5678",
    "주미 대한민국 대사관",
    "[QUESTION] 연락처를 알려주세요 [/QUESTION]",
]

for text in test_texts:
    encoded = tokenizer.encode(text)
    print(f"원본: {text}")
    print(f"토큰: {encoded.tokens}")
    print(f"ID: {encoded.ids}")
    print()
```

## 실행 방법

```bash
# 전제: data/samples.txt가 있어야 함
uv run python scripts/train_tokenizer.py
```

### 예상 출력

```
토크나이저 학습 시작...
입력 파일: data/samples.txt
어휘 크기: 24000
[00:00:30] Tokenizing ██████████████████████████████ 100%
토크나이저 저장 완료: data/tokenizer.json

--- 토크나이저 테스트 ---
원본: 안녕하세요. 연락처 정보입니다.
토큰: ['안녕하세요', '.', '연락처', '정보입니다', '.']
ID: [1234, 5, 2345, 3456, 5]

원본: TEL: 02-1234-5678
토큰: ['TEL:', '02-1234-5678']
ID: [100, 4567]
...
```

## 토큰화 → 바이너리 변환 (scripts/build_bin_dataset.py)

토크나이저 학습이 완료되면, 전체 텍스트를 토큰 ID 배열로 변환합니다.

```python
# scripts/build_bin_dataset.py
# 설명: 텍스트를 토큰 ID 배열로 변환하여 바이너리로 저장
# 입력: data/samples.txt, data/tokenizer.json
# 출력: data/train.bin, data/val.bin

import numpy as np
from tokenizers import Tokenizer

# ============================================
# 설정
# ============================================
TEXT_PATH = "data/samples.txt"      # 입력 텍스트
TOK_PATH = "data/tokenizer.json"    # 토크나이저

TRAIN_OUT = "data/train.bin"        # 학습용 출력
VAL_OUT = "data/val.bin"            # 검증용 출력

VAL_RATIO = 0.01  # 검증 데이터 비율 (1%)

# ============================================
# 토크나이저 로드
# ============================================
tokenizer = Tokenizer.from_file(TOK_PATH)
print(f"토크나이저 로드 완료: {TOK_PATH}")
print(f"어휘 크기: {tokenizer.get_vocab_size()}")

# ============================================
# 텍스트 로드 및 토큰화
# ============================================
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"텍스트 길이: {len(text):,} 문자")

# 토큰화
ids = tokenizer.encode(text).ids
print(f"토큰 수: {len(ids):,}")

# NumPy 배열로 변환
# vocab_size가 65535 이하면 uint16 사용 가능 (메모리 절약)
arr = np.array(ids, dtype=np.uint16)

# ============================================
# Train/Val 분할
# ============================================
n = len(arr)
n_val = int(n * VAL_RATIO)

train = arr[:-n_val]
val = arr[-n_val:]

# ============================================
# 바이너리로 저장
# ============================================
train.tofile(TRAIN_OUT)
val.tofile(VAL_OUT)

print(f"\n학습 데이터: {len(train):,} 토큰 → {TRAIN_OUT}")
print(f"검증 데이터: {len(val):,} 토큰 → {VAL_OUT}")
```

## 실행 방법

```bash
uv run python scripts/build_bin_dataset.py
```

### 예상 출력

```
토크나이저 로드 완료: data/tokenizer.json
어휘 크기: 24000
텍스트 길이: 50,000,000 문자
토큰 수: 15,000,000

학습 데이터: 14,850,000 토큰 → data/train.bin
검증 데이터: 150,000 토큰 → data/val.bin
```

## 토크나이저 사용 예시

### 인코딩 (텍스트 → 토큰 ID)

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer.json")

text = "안녕하세요. 연락처를 알려드립니다."
encoded = tokenizer.encode(text)

print(encoded.ids)     # [123, 456, 789, ...]
print(encoded.tokens)  # ['안녕하세요', '.', '연락처를', ...]
```

### 디코딩 (토큰 ID → 텍스트)

```python
ids = [123, 456, 789]
text = tokenizer.decode(ids)
print(text)  # "안녕하세요. 연락처를..."
```

## vocab_size 선택 가이드

| vocab_size | 장점 | 단점 | 권장 상황 |
|------------|------|------|-----------|
| 8,000 | 빠른 학습 | 한국어 표현력 부족 | 영어 위주 데이터 |
| 16,000 | 균형 잡힘 | - | 일반적인 경우 |
| 24,000 | 한국어 표현력 좋음 | 학습 약간 느림 | 한국어 + 전문용어 |
| 32,000 | 표현력 최고 | 학습 느림, 메모리 많이 사용 | 대용량 데이터 |

## 다음 단계

- [05-model-architecture.md](05-model-architecture.md) - GPT 모델 구조
