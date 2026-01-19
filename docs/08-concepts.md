# 핵심 개념

## 개요

LLM을 이해하기 위해 반드시 알아야 하는 9가지 핵심 개념을 설명합니다.

---

## 1. Tokenizer (토크나이저)

### 정의
텍스트를 정수 토큰 ID로 변환하는 컴포넌트

### 역할
```
"안녕하세요" → [1234, 567, 89] → 모델 입력
```

### 중요성
- vocab_size가 너무 작으면: 한국어가 깨짐 (글자 단위로 쪼개짐)
- vocab_size가 너무 크면: 학습이 어려워짐 (희소한 토큰 많음)
- 권장: 16,000 ~ 32,000

### 예시
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

# 인코딩
text = "안녕하세요"
ids = tokenizer.encode(text).ids  # [1234, 567]

# 디코딩
text = tokenizer.decode(ids)  # "안녕하세요"
```

---

## 2. Embedding (임베딩)

### 정의
토큰 ID를 고차원 벡터로 변환하는 테이블

### 역할
```
토큰 ID 1234 → [0.1, -0.3, 0.7, ..., 0.2]  (n_embd 차원)
```

### 왜 필요한가?
- 정수 ID는 의미 정보가 없음
- 벡터 공간에서는 의미적 유사성 표현 가능
- 예: "개"와 "강아지"의 벡터가 가까워짐

### 코드
```python
import torch.nn as nn

# vocab_size개의 토큰, 각각 n_embd 차원 벡터
embedding = nn.Embedding(vocab_size=24000, embedding_dim=384)

# 토큰 ID → 벡터
token_id = torch.tensor([1234])
vector = embedding(token_id)  # shape: (1, 384)
```

---

## 3. Positional Encoding/Embedding (위치 인코딩)

### 정의
토큰의 순서 정보를 모델에 전달하는 방법

### 왜 필요한가?
Transformer는 기본적으로 순서를 모름 (Self-Attention은 순서 무관)

```
"고양이가 개를 쫓았다" vs "개가 고양이를 쫓았다"
→ 위치 정보 없이는 같은 토큰들로 인식
```

### GPT의 방식 (Learnable Position Embedding)
```python
# 각 위치(0, 1, 2, ...)마다 학습 가능한 벡터
pos_embedding = nn.Embedding(block_size, n_embd)

# 최종 입력 = 토큰 임베딩 + 위치 임베딩
x = token_embedding + pos_embedding
```

---

## 4. Self-Attention (자기 어텐션)

### 정의
문장 내에서 "어떤 단어가 어떤 단어에 주목해야 하는지"를 학습

### 핵심 아이디어
```
"그 고양이는 매우 귀여웠다. 그것은 집에서 잤다."

"그것"이 무엇을 가리키는지 알려면?
→ "그것"은 "고양이"에 강하게 어텐션
```

### 수학적 표현
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q (Query): "내가 찾고 싶은 것"
K (Key): "나를 설명하는 키워드"
V (Value): "내가 가진 정보"
```

### Causal (인과적) Self-Attention
GPT는 **미래 토큰을 볼 수 없음** (마스킹)

```
"나는 밥을 [먹었다]"
→ "먹었다" 예측 시, "나는 밥을"만 참조 가능
```

---

## 5. Feed Forward Network (FFN)

### 정의
Attention 후에 정보를 비선형 변환하는 2층 MLP

### 구조
```
입력 (n_embd)
  → Linear (n_embd → 4*n_embd)
  → GELU 활성화
  → Linear (4*n_embd → n_embd)
  → 출력 (n_embd)
```

### 역할
- Attention은 "어디를 볼지" 결정
- FFN은 "본 정보를 어떻게 변환할지" 결정
- 표현력 증가

### 코드
```python
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)  # 활성화 함수
        x = self.proj(x)
        return x
```

---

## 6. Residual Connection + LayerNorm

### Residual Connection (잔차 연결)

```
출력 = 입력 + 레이어(입력)
```

#### 왜 필요한가?
- 깊은 네트워크에서 gradient 소실 방지
- "입력 정보를 그대로 전달" + "새로운 정보 추가"

### LayerNorm (레이어 정규화)

각 샘플 내에서 평균 0, 분산 1로 정규화

#### 왜 필요한가?
- 학습 안정화
- 각 레이어의 출력 스케일 일정하게 유지

### Pre-LayerNorm vs Post-LayerNorm

```
# Post-LN (원래 Transformer)
x = x + Attention(x)
x = LayerNorm(x)

# Pre-LN (GPT-2, 더 안정적)
x = x + Attention(LayerNorm(x))
```

JAI는 **Pre-LayerNorm** 사용 (GPT-2 스타일)

---

## 7. Next-token Prediction (다음 토큰 예측)

### 정의
GPT의 유일한 학습 목표

```
입력: "나는 밥을"
목표: "먹었다" 예측
```

### 손실 함수
**Cross-Entropy Loss**

```python
loss = F.cross_entropy(predicted_logits, target_token)
```

### 자기 지도 학습 (Self-supervised)
- 라벨링 불필요
- 텍스트 자체가 학습 데이터
- "다음 단어 맞추기"만으로 언어 이해

### 왜 이게 작동하는가?
다음 단어를 예측하려면:
- 문법을 이해해야 함
- 문맥을 파악해야 함
- 세상 지식이 필요함

→ 자연스럽게 언어 능력 습득

---

## 8. Sampling (샘플링)

### 정의
모델 출력(확률 분포)에서 다음 토큰을 선택하는 방법

### Greedy Decoding
가장 확률 높은 토큰 선택
```python
next_token = argmax(probs)
```
- 결정적 (같은 입력 → 같은 출력)
- 반복적이고 지루할 수 있음

### Temperature Sampling
확률 분포를 조절
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```

| Temperature | 효과 |
|-------------|------|
| < 1.0 | 확률 분포 뾰족 (보수적) |
| = 1.0 | 원래 분포 |
| > 1.0 | 확률 분포 평평 (다양) |

### Top-K Sampling
상위 K개 토큰만 고려
```python
top_k_probs = top_k(probs, k=50)
next_token = sample(top_k_probs)
```

### Top-P (Nucleus) Sampling
누적 확률 P까지의 토큰만 고려
```python
# 확률 높은 순으로 정렬 후
# 누적 확률이 P(예: 0.9)가 될 때까지의 토큰만 사용
```

---

## 9. 데이터 포맷이 곧 모델 능력

### 핵심 원리

> **모델은 학습 데이터의 패턴을 따라한다**

```
학습 데이터 형식 → 모델 출력 형식
```

### 예시

| 학습 데이터 | 모델 능력 |
|------------|----------|
| 요약문이 많음 | 요약 잘함 |
| 체크리스트 많음 | 체크리스트 생성 |
| Q&A 형식 | 질문에 답변 |
| 구인/구직 정보 | 구인/구직 형식 출력 |

### JAI의 데이터 전략

```
[QUESTION]
질문
[/QUESTION]

[ANSWER]
요약:
- ...

체크리스트:
- ...

구인 정보:
- 회사: ...
- 포지션: ...

상세 설명:
...
[/ANSWER]
```

이 형식으로 학습하면 → 모델도 이 형식으로 출력

### 중요한 교훈

```
파인튜닝 없이도, 데이터 포맷만 잘 설계하면
원하는 출력 형식을 얻을 수 있다.
```

---

## 개념 간 관계도

```
텍스트
  ↓
[Tokenizer] → 토큰 ID
  ↓
[Embedding] → 토큰 벡터
  ↓
  + [Positional Embedding] → 위치 정보 추가
  ↓
┌─────────────────────────────┐
│  Transformer Block (×N)    │
│  ┌─────────────────────┐   │
│  │ [Self-Attention]    │   │ ← 어디를 볼지
│  │ + Residual + LN     │   │
│  └──────────┬──────────┘   │
│             ↓              │
│  ┌─────────────────────┐   │
│  │ [Feed Forward]      │   │ ← 정보 변환
│  │ + Residual + LN     │   │
│  └──────────┬──────────┘   │
└─────────────┼──────────────┘
              ↓
[Linear Head] → 로짓 (vocab_size)
              ↓
[Sampling] → 다음 토큰
              ↓
[Tokenizer.decode] → 텍스트
```

## 다음 단계

- [09-tips.md](09-tips.md) - 팁과 트러블슈팅
