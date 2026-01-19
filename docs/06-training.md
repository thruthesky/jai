# 모델 학습

## 개요

GPT 모델의 학습 목표는 단 하나입니다:

> **"앞의 토큰들을 보고 다음 토큰을 맞춰라" (Next-token prediction)**

이 단순한 목표로 언어 모델이 텍스트를 생성하는 능력을 얻습니다.

## 학습 설정 (Config)

```python
from dataclasses import dataclass

@dataclass
class CFG:
    """학습 설정"""
    # 파일 경로
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tok_path: str = "data/tokenizer.json"
    out_dir: str = "checkpoints"

    # 모델 설정
    block_size: int = 256      # 컨텍스트 길이
    n_layer: int = 6           # Transformer 레이어 수
    n_head: int = 6            # Attention Head 수
    n_embd: int = 384          # 임베딩 차원
    dropout: float = 0.1       # 드롭아웃 비율

    # 학습 설정
    batch_size: int = 16       # 배치 크기
    lr: float = 3e-4           # 학습률
    max_steps: int = 20000     # 총 학습 스텝
    eval_interval: int = 500   # 평가 간격
    eval_iters: int = 100      # 평가 시 반복 횟수
    grad_clip: float = 1.0     # Gradient Clipping

    # 샘플링 설정 (디버그용)
    sample_every_eval: bool = True
    sample_max_new_tokens: int = 250
    temperature: float = 0.9
    top_k: int = 50
```

## 전체 학습 스크립트 (scripts/train_gpt.py)

```python
# scripts/train_gpt.py
# 설명: GPT 모델을 처음부터 학습하는 완전한 스크립트

import os
# MPS fallback 설정 (torch import 전에 반드시 설정)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import math
import time
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

# ============================================
# 설정
# ============================================
@dataclass
class CFG:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    tok_path: str = "data/tokenizer.json"
    out_dir: str = "checkpoints"

    # 모델
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    # 학습
    batch_size: int = 16
    lr: float = 3e-4
    max_steps: int = 20000
    eval_interval: int = 500
    eval_iters: int = 100
    grad_clip: float = 1.0

    # 샘플링
    sample_every_eval: bool = True
    sample_max_new_tokens: int = 250
    temperature: float = 0.9
    top_k: int = 50

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# ============================================
# 디바이스 설정
# ============================================
def get_device():
    """최적의 디바이스 자동 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"사용 디바이스: {device}")

# ============================================
# 토크나이저 로드
# ============================================
tokenizer = Tokenizer.from_file(cfg.tok_path)
vocab_size = tokenizer.get_vocab_size()
print(f"어휘 크기: {vocab_size}")

# ============================================
# 데이터 로더
# ============================================
# 메모리 매핑으로 대용량 데이터 효율적으로 처리
train_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")

print(f"학습 데이터: {len(train_data):,} 토큰")
print(f"검증 데이터: {len(val_data):,} 토큰")

def get_batch(split: str):
    """
    랜덤 배치 생성

    Args:
        split: "train" 또는 "val"

    Returns:
        x: 입력 토큰 (batch_size, block_size)
        y: 타겟 토큰 (batch_size, block_size) - x를 한 칸 shift
    """
    data = train_data if split == "train" else val_data

    # 랜덤 시작 위치 선택
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))

    # 입력과 타겟 생성
    x = torch.stack([
        torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)

# ============================================
# 모델 정의 (이전 섹션 참조)
# ============================================
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
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for b in self.blocks:
            x = b(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# ============================================
# 모델 & 옵티마이저 초기화
# ============================================
model = GPT(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

# 파라미터 수 출력
n_params = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터 수: {n_params:,}")

# ============================================
# 체크포인트 로드 (있으면)
# ============================================
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
start_step = 0

if os.path.exists(ckpt_path):
    print(f"체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    start_step = ckpt.get("step", 0)
    print(f"스텝 {start_step}부터 재개")

# ============================================
# 평가 함수
# ============================================
@torch.no_grad()
def estimate_loss():
    """학습/검증 손실 추정"""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ============================================
# 샘플 생성 함수
# ============================================
def decode_ids(ids):
    """토큰 ID를 텍스트로 변환"""
    return tokenizer.decode(ids)

def quick_sample():
    """학습 중 샘플 생성"""
    prompt = """[QUESTION]
전 세계 대사관 연락처 정보를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
-"""
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor(enc, dtype=torch.long, device=device)[None, :]
    y = model.generate(
        x,
        max_new_tokens=cfg.sample_max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
    )
    out = decode_ids(y[0].tolist())
    return out

# ============================================
# 학습 루프
# ============================================
print("학습 시작...")
t0 = time.time()

pbar = tqdm(range(start_step, cfg.max_steps))
for step in pbar:
    # 배치 가져오기
    x, y = get_batch("train")

    # 순전파
    logits, loss = model(x, y)

    # 역전파
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 가중치 업데이트
    optimizer.step()

    # 진행률 업데이트
    if step % 50 == 0:
        pbar.set_description(f"step {step} loss {loss.item():.4f}")

    # 평가 & 체크포인트 저장
    if step > 0 and step % cfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"\nstep {step} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")

        # 체크포인트 저장
        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
            "cfg": cfg.__dict__,
        }, ckpt_path)
        print(f"체크포인트 저장: {ckpt_path}")

        # 샘플 생성
        if cfg.sample_every_eval:
            print("\n--- 샘플 생성 ---")
            print(quick_sample()[:2500])
            print("-" * 50 + "\n")

print(f"학습 완료. 소요 시간: {time.time() - t0:.1f}초")
```

## 실행 방법

```bash
uv run python scripts/train_gpt.py
```

## 학습 모니터링

### 예상 출력

```
사용 디바이스: mps
어휘 크기: 24000
학습 데이터: 14,850,000 토큰
검증 데이터: 150,000 토큰
모델 파라미터 수: 28,123,456
학습 시작...
step 0 loss 10.1234: 100%|██████████████| 50/20000
step 50 loss 8.5432: 100%|██████████████| 100/20000
...
step 500 train_loss=5.2341 val_loss=5.3456
체크포인트 저장: checkpoints/ckpt.pt

--- 샘플 생성 ---
[QUESTION]
전 세계 대사관 연락처 정보를 정리해줘.
[/QUESTION]

[ANSWER]
요약:
- 대사관 연락처는 국가별로 다름...
--------------------------------------------------
```

### 손실(Loss) 변화 해석

| 손실 값 | 의미 |
|---------|------|
| 10+ | 초기 상태, 랜덤 |
| 5~7 | 문법 패턴 학습 시작 |
| 3~5 | 형식 학습, 연락처 패턴 인식 |
| 2~3 | 의미 있는 텍스트 생성 |
| <2 | 좋은 품질 (과적합 주의) |

## 학습 팁

### 1. 손실이 줄지 않을 때

```python
# 학습률 낮추기
cfg.lr = 1e-4  # 3e-4 → 1e-4

# 배치 크기 늘리기 (메모리 허용 시)
cfg.batch_size = 32
```

### 2. 메모리 부족 (OOM)

```python
# 배치 크기 줄이기
cfg.batch_size = 8  # 16 → 8

# 컨텍스트 길이 줄이기
cfg.block_size = 128  # 256 → 128

# 모델 크기 줄이기
cfg.n_layer = 4   # 6 → 4
cfg.n_embd = 256  # 384 → 256
```

### 3. 학습 중단 후 재개

체크포인트가 자동으로 저장되므로, 스크립트를 다시 실행하면 자동으로 재개됩니다.

```bash
uv run python scripts/train_gpt.py
```

## 다음 단계

- [07-generation.md](07-generation.md) - 텍스트 생성
