# 텍스트 생성

## 개요

학습된 모델로 텍스트를 생성합니다.
CAI는 **요약/정리형 연락처 정보**를 생성하도록 설계되었습니다.

## 생성 원리

GPT는 **Autoregressive** 방식으로 텍스트를 생성합니다:

```
입력: "안녕"
→ 모델이 다음 토큰 예측: "하"
→ 입력 업데이트: "안녕하"
→ 모델이 다음 토큰 예측: "세"
→ 입력 업데이트: "안녕하세"
→ ... 반복
```

## 프롬프트 형식 (중요!)

학습 데이터와 **동일한 형식**으로 프롬프트를 작성해야 좋은 결과가 나옵니다.

### 권장 프롬프트 템플릿

```
[QUESTION]
{질문 내용}
[/QUESTION]

[ANSWER]
요약:
-
```

`[ANSWER] 요약: -` 까지 입력하면, 모델이 자연스럽게 이어서 작성합니다.

## 생성 스크립트 (scripts/generate.py)

```python
# scripts/generate.py
# 설명: 학습된 모델로 텍스트 생성
# 입력: checkpoints/ckpt.pt, data/tokenizer.json
# 출력: 콘솔에 생성된 텍스트 출력

import torch
from tokenizers import Tokenizer

# ============================================
# 모델 설정 (학습 시와 동일해야 함!)
# ============================================
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1  # 추론 시에는 사용 안 됨 (eval 모드)

# ============================================
# 모델 정의 (학습 코드와 동일)
# ============================================
import math
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        텍스트 생성

        Args:
            idx: 시작 토큰 (B, T)
            max_new_tokens: 생성할 토큰 수
            temperature: 온도 (높을수록 다양함)
            top_k: Top-K 샘플링

        Returns:
            생성된 토큰 시퀀스
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# ============================================
# 디바이스 설정
# ============================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"사용 디바이스: {device}")

# ============================================
# 토크나이저 & 모델 로드
# ============================================
tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"어휘 크기: {vocab_size}")

model = GPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
).to(device)

# 체크포인트 로드
ckpt = torch.load("checkpoints/ckpt.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()  # 추론 모드
print("모델 로드 완료")

# ============================================
# 생성 함수
# ============================================
def generate_text(prompt, max_new_tokens=400, temperature=0.9, top_k=50):
    """
    프롬프트에 이어서 텍스트 생성

    Args:
        prompt: 시작 텍스트
        max_new_tokens: 생성할 최대 토큰 수
        temperature: 샘플링 온도
        top_k: Top-K 샘플링

    Returns:
        생성된 전체 텍스트
    """
    # 토큰화
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    # 생성
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    # 디코딩
    return tokenizer.decode(y[0].tolist())

# ============================================
# 예시 실행
# ============================================
if __name__ == "__main__":
    # 프롬프트 정의
    prompt = """[QUESTION]
미국 주요 도시의 한국 대사관/영사관 연락처를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-"""

    print("=" * 60)
    print("CAI (Contact AI) 텍스트 생성")
    print("=" * 60)
    print("\n[프롬프트]")
    print(prompt)
    print("\n[생성 결과]")
    print("-" * 60)

    result = generate_text(
        prompt,
        max_new_tokens=400,
        temperature=0.9,
        top_k=50
    )

    print(result[:4000])  # 최대 4000자까지 출력
    print("-" * 60)
```

## 실행 방법

```bash
uv run python scripts/generate.py
```

## 샘플링 파라미터 이해

### Temperature

출력의 **다양성/창의성**을 조절합니다.

| 값 | 효과 |
|----|------|
| 0.1~0.5 | 보수적, 예측 가능, 반복적 |
| 0.7~0.9 | 균형 잡힌 다양성 (권장) |
| 1.0+ | 창의적, 예측 불가, 가끔 엉뚱함 |

```python
# 보수적 (정확한 정보 원할 때)
generate_text(prompt, temperature=0.5)

# 창의적 (다양한 표현 원할 때)
generate_text(prompt, temperature=1.2)
```

### Top-K

상위 K개의 토큰만 샘플링 후보로 사용합니다.

| 값 | 효과 |
|----|------|
| 10~20 | 매우 보수적, 반복 위험 |
| 40~60 | 균형 (권장) |
| 100+ | 다양하지만 품질 저하 위험 |

```python
# 보수적
generate_text(prompt, top_k=20)

# 다양한
generate_text(prompt, top_k=100)
```

### 권장 조합

| 용도 | temperature | top_k |
|------|-------------|-------|
| 정확한 연락처 정보 | 0.7 | 30 |
| 일반적 사용 | 0.9 | 50 |
| 창의적 글쓰기 | 1.1 | 100 |

## 생성 결과 예시

### 입력 프롬프트

```
[QUESTION]
일본 도쿄 주재 한국 대사관 연락처와 업무 안내를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-
```

### 기대 출력

```
[ANSWER]
요약:
- 주일 대한민국 대사관은 도쿄 미나토구에 위치
- 영사과 업무는 별도 번호로 문의
- 평일 09:00-17:00 운영

체크리스트:
- 해야 할 일:
  - (1) 방문 전 전화 예약
  - (2) 필요 서류 사전 확인
- 준비물:
  - (1) 여권 원본
  - (2) 신청서
- 주의사항:
  - (1) 공휴일 휴무

연락처(공공정보):
- 주일 대한민국 대사관
  - TEL: +81-3-3452-7611
  - ADDR: 東京都港区南麻布1-2-5
  - WEB: https://overseas.mofa.go.kr/jp-ko/

상세 설명:
주일 대한민국 대사관은 도쿄 미나토구에 위치하고 있습니다...
[/ANSWER]
```

## 다음 단계

- [08-concepts.md](08-concepts.md) - 핵심 개념 설명
