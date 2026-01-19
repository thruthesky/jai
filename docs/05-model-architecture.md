# GPT 모델 아키텍처

## 개요

CAI는 **Decoder-only Transformer** 구조를 사용합니다.
이는 GPT, LLaMA, Claude 등 현대 LLM의 기본 구조입니다.

## 전체 구조도

```
입력 토큰 [I1, I2, I3, ..., In]
           ↓
    ┌──────────────┐
    │  Token       │  토큰 → 벡터 변환
    │  Embedding   │  (vocab_size, n_embd)
    └──────┬───────┘
           │
           + ← Position Embedding (위치 정보 추가)
           │
    ┌──────┴───────┐
    │   Dropout    │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │              │
    │   Block ×N   │  ← Transformer Block 반복
    │              │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  LayerNorm   │  최종 정규화
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  Linear      │  (n_embd → vocab_size)
    │  (Head)      │
    └──────┬───────┘
           │
           ↓
    출력 로짓 [O1, O2, O3, ..., On]
```

## Transformer Block 구조

```
입력 x
    │
    ├──────────────────┐
    │                  │ (Residual Connection)
    ↓                  │
┌────────────┐         │
│ LayerNorm  │         │
└─────┬──────┘         │
      │                │
      ↓                │
┌─────────────────┐    │
│ Causal Self     │    │
│ Attention       │    │
└─────┬───────────┘    │
      │                │
      + ←──────────────┘
      │
      ├──────────────────┐
      │                  │ (Residual Connection)
      ↓                  │
┌────────────┐           │
│ LayerNorm  │           │
└─────┬──────┘           │
      │                  │
      ↓                  │
┌─────────────────┐      │
│ Feed Forward    │      │
│ (MLP)           │      │
└─────┬───────────┘      │
      │                  │
      + ←────────────────┘
      │
      ↓
    출력
```

## 권장 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| vocab_size | 24,000 | 토크나이저와 맞춤 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention Head 수 |
| n_embd | 384 | 임베딩 차원 |
| dropout | 0.1 | 드롭아웃 비율 |

## 핵심 컴포넌트 코드

### 1. Causal Self-Attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Causal (인과적) Self-Attention
    - 현재 위치에서 미래의 토큰을 볼 수 없도록 마스킹
    - GPT의 핵심 메커니즘
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()

        # n_embd가 n_head로 나누어 떨어져야 함
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 각 헤드의 차원

        # Q, K, V를 한 번에 계산 (효율성)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)  # 출력 프로젝션

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal Mask (하삼각 행렬)
        # 미래 토큰을 마스킹하여 볼 수 없게 함
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # (배치, 시퀀스 길이, 임베딩 차원)

        # Q, K, V 계산
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # 각각 (B, T, C)

        # 멀티헤드로 reshape
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention Score 계산: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal Masking: 미래 위치를 -inf로 설정
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax → Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Value와 곱하기
        y = att @ v  # (B, n_head, T, head_dim)

        # 헤드 합치기
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 출력 프로젝션 + Dropout
        y = self.resid_drop(self.proj(y))

        return y
```

### 2. Feed Forward Network (MLP)

```python
class MLP(nn.Module):
    """
    Feed Forward Network
    - 2층 MLP with GELU 활성화
    - 중간 차원은 4배로 확장
    """
    def __init__(self, n_embd, dropout):
        super().__init__()

        # 4배 확장 후 다시 원래 크기로
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)      # 확장
        x = F.gelu(x)       # GELU 활성화
        x = self.proj(x)    # 축소
        x = self.drop(x)    # Dropout
        return x
```

### 3. Transformer Block

```python
class Block(nn.Module):
    """
    Transformer Block
    - Pre-LayerNorm 구조 (GPT-2 스타일)
    - Residual Connection 포함
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)  # Attention 전 정규화
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)  # MLP 전 정규화
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        # Residual + Self-Attention
        x = x + self.attn(self.ln1(x))
        # Residual + MLP
        x = x + self.mlp(self.ln2(x))
        return x
```

### 4. 전체 GPT 모델

```python
class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer)
    - Decoder-only 구조
    - 다음 토큰 예측 (Next-token prediction)
    """
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()

        self.block_size = block_size

        # 임베딩 레이어
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # 토큰 임베딩
        self.pos_emb = nn.Embedding(block_size, n_embd)  # 위치 임베딩
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])

        # 최종 레이어
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # 출력 헤드

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화 (GPT-2 스타일)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        순전파

        Args:
            idx: 입력 토큰 ID (B, T)
            targets: 타겟 토큰 ID (B, T), 학습 시에만 사용

        Returns:
            logits: 출력 로짓 (B, T, vocab_size)
            loss: CrossEntropy 손실 (targets가 있을 때만)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"시퀀스 길이 {T}가 block_size {self.block_size}를 초과"

        # 위치 인덱스 생성
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        # 토큰 임베딩 + 위치 임베딩
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, n_embd)
        x = self.drop(x)

        # Transformer Blocks 통과
        for block in self.blocks:
            x = block(x)

        # 최종 정규화 + 출력 헤드
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        # 손실 계산 (학습 시)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1)                     # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        텍스트 생성 (Autoregressive)

        Args:
            idx: 시작 토큰 ID (B, T)
            max_new_tokens: 생성할 토큰 수
            temperature: 샘플링 온도 (높을수록 다양함)
            top_k: Top-K 샘플링 (None이면 전체 vocab 사용)

        Returns:
            idx: 생성된 토큰 시퀀스 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 컨텍스트 길이 제한
            idx_cond = idx[:, -self.block_size:]

            # 순전파
            logits, _ = self(idx_cond)

            # 마지막 토큰의 로짓만 사용
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Top-K 샘플링
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # 확률 분포로 변환
            probs = F.softmax(logits, dim=-1)

            # 샘플링
            next_id = torch.multinomial(probs, num_samples=1)

            # 시퀀스에 추가
            idx = torch.cat((idx, next_id), dim=1)

        return idx
```

## 모델 파라미터 수 계산

```python
def count_parameters(model):
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 예시
model = GPT(
    vocab_size=24000,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1
)

params = count_parameters(model)
print(f"총 파라미터 수: {params:,}")  # 약 28M
```

### 파라미터 수

기본 설정(6 layer, 384 dim)으로 약 **28M** 파라미터입니다.

## 다음 단계

- [06-training.md](06-training.md) - 모델 학습 방법
