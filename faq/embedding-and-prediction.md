# 토큰화, 임베딩, 그리고 다음 토큰 예측

GPT 모델이 텍스트를 어떻게 처리하고 다음 단어를 예측하는지 상세하게 설명합니다.

---

## 자주 묻는 질문

### Q1: 토큰화 한 다음 → 임베딩 절차를 거치나요?

**답변: 예, 맞습니다.**

토큰화된 ID는 임베딩 레이어를 통해 벡터로 변환됩니다.

### Q2: 임베딩된 데이터를 벡터 검색을 통해서 다음 토큰(단어)를 예측하나요?

**답변: 아니요, 다릅니다.**

GPT의 다음 토큰 예측은 벡터 검색(유사도 검색)이 아닙니다. 이 부분은 흔한 오해입니다.

---

## 1. GPT의 전체 처리 흐름

```
텍스트 입력
    ↓
[토큰화] → 토큰 ID 배열
    ↓
[임베딩] → 벡터 배열 (모델 내부)
    ↓
[Transformer 블록 × N] → 문맥을 이해한 벡터
    ↓
[출력 헤드] → vocab_size 크기의 확률 분포
    ↓
[샘플링] → 다음 토큰 선택
    ↓
텍스트 출력
```

---

## 2. 토큰화 → 임베딩: 상세 설명

### 토큰화 (Tokenization)

텍스트를 작은 조각(토큰)으로 나누고 각 조각에 정수 ID를 부여합니다.

```python
# 입력 텍스트
text = "서울에서 React 개발자 채용"

# 토큰화 결과
tokens = ["서울", "에서", " ", "React", " ", "개발자", " ", "채용"]
token_ids = [1523, 892, 3, 4521, 3, 2847, 3, 1956]
```

### 임베딩 (Embedding)

각 토큰 ID를 고차원 벡터로 변환합니다. 이 벡터는 토큰의 "의미"를 담고 있습니다.

```python
# 토큰 ID
token_ids = [1523, 892, 4521]  # "서울", "에서", "React"

# 임베딩 변환 (n_embd = 384 차원)
embeddings = [
    [0.12, -0.34, 0.56, ..., 0.78],   # "서울" 벡터 (384개 숫자)
    [0.23, 0.45, -0.67, ..., 0.89],   # "에서" 벡터 (384개 숫자)
    [0.34, -0.56, 0.78, ..., -0.12],  # "React" 벡터 (384개 숫자)
]
```

### 임베딩 테이블

임베딩은 거대한 "룩업 테이블"입니다.

```
vocab_size = 24,000 (토큰 개수)
n_embd = 384 (벡터 차원)

임베딩 테이블 크기: 24,000 × 384 = 9,216,000개의 숫자
```

```python
import torch.nn as nn

# 임베딩 레이어 생성
embedding = nn.Embedding(num_embeddings=24000, embedding_dim=384)

# 토큰 ID → 벡터 변환
token_id = torch.tensor([1523])  # "서울"
vector = embedding(token_id)      # shape: (1, 384)
```

### 중요: 임베딩은 모델 내부에 포함됨

**별도로 임베딩을 수행할 필요가 없습니다.** GPT 모델이 토큰 ID를 입력받으면 내부에서 자동으로 임베딩을 수행합니다.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 토큰 임베딩 (모델 내부에 포함)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 위치 임베딩 (모델 내부에 포함)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        # ...

    def forward(self, idx):
        # idx: 토큰 ID 배열
        tok_emb = self.tok_emb(idx)  # 자동으로 임베딩 수행
        pos_emb = self.pos_emb(torch.arange(idx.size(1)))
        x = tok_emb + pos_emb  # 토큰 + 위치 임베딩
        # ...
```

---

## 2.1 임베딩 육하원칙: 누가, 무엇을, 언제, 어디서, 왜, 어떻게

### 누가 (Who): GPT 모델이 직접 수행

```python
# scripts/train_gpt.py에서 직접 구현하는 GPT 클래스
class GPT(nn.Module):
    def __init__(self, config):
        # GPT 모델이 임베딩 레이어를 소유
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
```

- **외부 라이브러리 사용 안 함**: Hugging Face, OpenAI API 등 사용하지 않음
- **PyTorch 기본 레이어 사용**: `nn.Embedding`은 PyTorch가 제공하는 기본 빌딩 블록
- **직접 구현**: JAI 프로젝트의 목표는 "처음부터 직접 구현"

### 무엇을 (What): 토큰 ID → 고차원 벡터 변환

```
입력: 정수 ID (예: 1523)
출력: 384차원 벡터 (예: [0.12, -0.34, 0.56, ..., 0.78])
```

| 변환 전 | 변환 후 |
|--------|--------|
| 정수 1개 | 실수 384개 |
| 의미 없음 | 의미를 담은 벡터 |
| 연산 불가 | 행렬 연산 가능 |

### 언제 (When): 토큰화 직후, Transformer 블록 통과 전

```
1. 텍스트 입력: "서울에서 React"
2. 토큰화: [1523, 892, 4521]
3. ⭐ 임베딩: [[0.12, ...], [0.23, ...], [0.34, ...]]  ← 여기서 수행
4. 위치 임베딩 추가
5. Transformer 블록 통과
6. 출력 헤드
7. 샘플링
```

### 어디서 (Where): GPT 클래스 내부의 forward() 함수

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 임베딩 레이어 정의 (여기서 테이블 생성)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        B, T = idx.size()  # 배치 크기, 시퀀스 길이

        # ⭐ 임베딩 수행 (여기서 실행)
        tok_emb = self.tok_emb(idx)                      # (B, T, n_embd)
        pos_emb = self.pos_emb(torch.arange(T))          # (T, n_embd)
        x = tok_emb + pos_emb                            # (B, T, n_embd)

        # Transformer 블록 통과
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
```

### 왜 (Why): 정수는 의미가 없고, 벡터는 의미를 담을 수 있음

#### 정수 ID의 한계

```
토큰 ID 1523 = "서울"
토큰 ID 1524 = "부산"
토큰 ID 1525 = "apple"

1523과 1524는 숫자상 가깝지만, "서울"과 "부산"의 유사성을 반영하지 않음
1524와 1525도 숫자상 가깝지만, "부산"과 "apple"은 전혀 다른 의미
```

#### 벡터의 장점

```
"서울" 벡터: [0.8, 0.2, 0.9, ...]  (한국, 도시, 수도 등의 의미)
"부산" 벡터: [0.7, 0.3, 0.85, ...] (한국, 도시, 항구 등의 의미)
"apple" 벡터: [-0.5, 0.9, -0.2, ...] (과일, 영어, 음식 등의 의미)

→ "서울"과 "부산"의 벡터가 가까움 (코사인 유사도 높음)
→ "서울"과 "apple"의 벡터가 멀음 (코사인 유사도 낮음)
```

### 어떻게 (How): 룩업 테이블 + 학습

#### 1단계: 임베딩 테이블 초기화

```python
# 모델 생성 시 랜덤 값으로 초기화
self.tok_emb = nn.Embedding(24000, 384)

# 내부적으로 이런 테이블이 생성됨 (24000 × 384 크기)
# 모든 값이 랜덤
embedding_table = [
    [0.01, -0.02, 0.03, ..., 0.01],   # 토큰 0번 벡터
    [-0.01, 0.02, -0.03, ..., -0.01], # 토큰 1번 벡터
    [0.02, -0.01, 0.02, ..., 0.02],   # 토큰 2번 벡터
    ...
    [0.01, 0.01, -0.02, ..., 0.03],   # 토큰 23999번 벡터
]
```

#### 2단계: 룩업 (단순 인덱싱)

```python
# 토큰 ID로 테이블에서 해당 행을 가져옴
token_id = 1523  # "서울"
vector = embedding_table[1523]  # 1523번째 행 반환

# 이게 임베딩의 전부! 단순한 테이블 조회
```

```
nn.Embedding의 동작 = 배열 인덱싱
embedding(torch.tensor([1523])) == embedding_table[1523]
```

#### 3단계: 학습으로 테이블 업데이트

```python
# 학습 전: 랜덤 값
"서울" 벡터: [0.01, -0.02, 0.03, ...]  # 의미 없음
"부산" 벡터: [-0.05, 0.08, -0.01, ...] # 의미 없음

# 학습 후: 의미 있는 값
"서울" 벡터: [0.82, 0.21, 0.93, ...]   # 한국 도시의 의미
"부산" 벡터: [0.78, 0.25, 0.88, ...]   # 한국 도시의 의미 (서울과 유사)
```

**학습 과정:**
1. 모델이 "나는 서울에서"의 다음 토큰을 예측
2. 정답과 비교하여 손실(loss) 계산
3. 역전파(backpropagation)로 기울기 계산
4. 임베딩 테이블의 값도 함께 업데이트됨
5. 수천만 번 반복하면 의미 있는 벡터가 됨

#### 전체 흐름 요약

```
[모델 생성]
    ↓
nn.Embedding(24000, 384) → 랜덤 테이블 생성 (24000 × 384)
    ↓
[학습 시작]
    ↓
토큰 ID 입력 → 테이블에서 벡터 조회 (룩업)
    ↓
Transformer 연산 → 예측 → 손실 계산
    ↓
역전파 → 임베딩 테이블 값 업데이트
    ↓
[수천만 번 반복]
    ↓
의미 있는 임베딩 테이블 완성
    ↓
[체크포인트 저장] → 학습된 임베딩도 함께 저장됨
```

### 정리: 육하원칙 요약표

| 질문 | 답변 |
|------|------|
| **누가** | GPT 모델 (train_gpt.py에서 직접 구현) |
| **무엇을** | 토큰 ID → 384차원 벡터 변환 |
| **언제** | 토큰화 직후, Transformer 블록 통과 전 |
| **어디서** | GPT.forward() 함수 내부 |
| **왜** | 정수는 의미 없음, 벡터는 의미 담을 수 있음 |
| **어떻게** | nn.Embedding 룩업 테이블 + 학습으로 값 업데이트 |

### Q&A

**Q: 별도의 임베딩 모델이 필요한가요?**

아니요. `nn.Embedding`이 GPT 클래스 안에 이미 정의되어 있습니다.

**Q: 임베딩 값은 어디서 오나요?**

처음엔 랜덤, 학습하면서 의미 있는 값으로 업데이트됩니다.

**Q: 직접 임베딩 코드를 작성해야 하나요?**

`train_gpt.py`에서 GPT 클래스를 정의할 때 `nn.Embedding`을 포함시키면 끝입니다. 나머지는 PyTorch가 자동으로 처리합니다.

---

## 3. 벡터 검색 vs GPT 예측: 핵심 차이

### 흔한 오해

> "GPT가 임베딩 벡터들 중에서 가장 비슷한 벡터를 찾아서 다음 단어를 예측한다"

**이것은 틀린 이해입니다.**

### 벡터 검색 (RAG, 유사도 검색)

벡터 검색은 **외부 문서/지식을 검색**할 때 사용합니다.

```
질문 벡터 → 문서 벡터들과 코사인 유사도 비교 → 가장 유사한 문서 반환
```

```python
# 벡터 검색 예시 (RAG)
from sklearn.metrics.pairwise import cosine_similarity

query_vector = embed("서울 개발자 채용")  # 질문 벡터
doc_vectors = [embed(doc) for doc in documents]  # 문서 벡터들

# 유사도 계산
similarities = cosine_similarity([query_vector], doc_vectors)
most_similar_idx = similarities.argmax()  # 가장 유사한 문서 인덱스
```

### GPT의 다음 토큰 예측

GPT는 **벡터 검색이 아닌 수학적 연산**으로 다음 토큰을 예측합니다.

```
임베딩 벡터 → Transformer 연산 → 확률 분포 → 샘플링
```

```python
# GPT 다음 토큰 예측 (실제 방식)
def predict_next_token(model, input_ids):
    # 1. 임베딩
    x = model.tok_emb(input_ids) + model.pos_emb(...)

    # 2. Transformer 블록 통과 (Self-Attention + FFN)
    for block in model.blocks:
        x = block(x)  # 행렬 곱셈, Attention 연산

    # 3. 출력 헤드: 벡터 → vocab_size 확률
    logits = model.lm_head(x)  # shape: (batch, seq_len, vocab_size)

    # 4. 마지막 위치의 확률 분포
    probs = F.softmax(logits[:, -1, :], dim=-1)  # shape: (batch, vocab_size)

    # 5. 샘플링
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### 비교 표

| | 벡터 검색 (RAG) | GPT 다음 토큰 예측 |
|---|---|---|
| **방식** | 코사인 유사도로 "가장 비슷한" 벡터 찾기 | Transformer 연산 후 확률 분포 생성 |
| **용도** | 외부 문서/지식 검색 | 언어 모델 내부 예측 |
| **입력** | 질문 벡터, 문서 벡터들 | 이전 토큰들의 임베딩 |
| **출력** | 유사한 문서/벡터 | vocab_size 크기의 확률 분포 |
| **연산** | 유사도 계산 (내적/코사인) | Self-Attention, FFN, 행렬 곱셈 |

---

## 4. GPT 예측 과정 상세 설명

### 예시: "나는 밥을" 다음 토큰 예측

```
입력: "나는 밥을"
```

#### 단계 1: 토큰화

```
"나는 밥을" → [나는, 밥을] → [2341, 5678]
```

#### 단계 2: 임베딩

```python
token_ids = [2341, 5678]

# 토큰 임베딩
tok_emb = [
    [0.1, -0.2, 0.3, ...],  # "나는" (384차원)
    [0.4, 0.5, -0.6, ...],  # "밥을" (384차원)
]

# 위치 임베딩 추가
pos_emb = [
    [0.01, 0.02, -0.01, ...],  # 위치 0
    [0.02, -0.01, 0.03, ...],  # 위치 1
]

# 최종 입력 = 토큰 임베딩 + 위치 임베딩
x = tok_emb + pos_emb
```

#### 단계 3: Transformer 블록 통과

```
x → [Block 1] → [Block 2] → ... → [Block 6] → x'

각 블록에서:
- LayerNorm
- Self-Attention (어떤 토큰이 어떤 토큰에 주목할지)
- Residual Connection
- LayerNorm
- FFN (Feed Forward Network)
- Residual Connection
```

**Self-Attention 예시:**
```
"나는"이 "밥을"에 얼마나 주목할지 → Attention Score 계산
"밥을"이 "나는"에 얼마나 주목할지 → Attention Score 계산
```

#### 단계 4: 출력 헤드 (Linear Layer)

```python
# Transformer 출력 (384차원)
x_last = x'[-1]  # 마지막 토큰 "밥을"의 출력 벡터

# 출력 헤드: 384차원 → 24000차원 (vocab_size)
logits = linear(x_last)  # [0.2, -1.5, 3.2, ..., 0.8]  (24000개)
```

#### 단계 5: 확률 분포 생성

```python
# Softmax로 확률 변환
probs = softmax(logits)

# 결과 (예시)
# 토큰 ID 7892 ("먹었다"): 0.35
# 토큰 ID 7893 ("먹는다"): 0.20
# 토큰 ID 4521 ("좋아한다"): 0.05
# ...
# 총합: 1.0
```

#### 단계 6: 샘플링

```python
# 확률에 따라 토큰 선택
# - Greedy: 가장 높은 확률 선택 → "먹었다"
# - Temperature: 확률 분포 조절 후 샘플링
# - Top-K: 상위 K개 중에서 샘플링

next_token_id = sample(probs)  # 7892 ("먹었다")
```

#### 최종 결과

```
입력: "나는 밥을"
출력: "먹었다"

전체: "나는 밥을 먹었다"
```

---

## 5. 왜 벡터 검색이 아닌가?

### 벡터 검색의 한계

벡터 검색은 **"가장 비슷한 것"**을 찾습니다.

```
"나는 밥을" → 가장 비슷한 벡터 찾기 → "너는 밥을"?
```

이 방식으로는 **다음에 올 단어**를 예측할 수 없습니다.

### GPT의 강점

GPT는 **"다음에 올 확률이 높은 것"**을 계산합니다.

```
"나는 밥을" → 다음에 올 확률 계산 →
    "먹었다" (35%)
    "먹는다" (20%)
    "좋아한다" (5%)
    ...
```

이 차이가 핵심입니다:
- **유사도**: 비슷한 것 찾기 (검색)
- **확률**: 다음에 올 것 예측 (생성)

---

## 6. Self-Attention이 하는 일

벡터 검색과 혼동하기 쉬운 부분이 Self-Attention입니다.

### Self-Attention ≠ 벡터 검색

| | Self-Attention | 벡터 검색 |
|---|---|---|
| 목적 | 문장 내 단어 간 관계 학습 | 외부 문서 찾기 |
| 범위 | 입력 문장 내부 | 외부 문서 컬렉션 |
| 결과 | 문맥을 반영한 벡터 | 가장 유사한 문서 |

### Self-Attention 예시

```
입력: "그 고양이는 매우 귀여웠다. 그것은 집에서 잤다."

Self-Attention이 학습하는 것:
- "그것"은 "고양이"를 가리킴 → 높은 Attention Score
- "그것"과 "집"은 관계 약함 → 낮은 Attention Score
```

Self-Attention은 **문장 내에서 어떤 단어가 어떤 단어에 주목해야 하는지**를 학습합니다. 이것은 벡터 검색과 완전히 다른 연산입니다.

---

## 7. 정리

### 핵심 요약

| 질문 | 답변 |
|------|------|
| 토큰화 → 임베딩 거치나요? | ✅ **예**, 모델 내부에서 자동 수행 |
| 벡터 검색으로 다음 토큰 예측? | ❌ **아니요**, Transformer 연산 + 확률 분포 |

### GPT의 작동 방식

```
토큰 ID → 임베딩 → Transformer 연산 → 확률 분포 → 샘플링 → 다음 토큰
```

1. **임베딩**: 토큰 ID를 벡터로 변환 (룩업 테이블)
2. **Transformer**: Self-Attention + FFN으로 문맥 이해
3. **출력 헤드**: 벡터를 vocab_size 확률 분포로 변환
4. **샘플링**: 확률에 따라 다음 토큰 선택

### 벡터 검색은 언제 사용?

벡터 검색은 **RAG(검색 증강 생성)**에서 사용됩니다.

```
사용자 질문 → 벡터 검색으로 관련 문서 찾기 → GPT에게 전달 → 답변 생성
```

RAG는 GPT와 벡터 검색을 **결합**한 시스템이지, GPT 내부 작동 방식이 아닙니다.

---

## 참고

- [핵심 개념 9가지](core-concepts.md) - Self-Attention, Embedding 등
- [데이터 흐름](data-flow.md) - 전체 파이프라인
- [모델 구조](../docs/05-model-architecture.md) - GPT 아키텍처 상세
