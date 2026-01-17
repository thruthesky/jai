# 팁과 트러블슈팅

## 개요

CAI 개발 중 자주 발생하는 문제와 해결 방법, 그리고 성능 최적화 팁을 정리합니다.

---

## 환경 관련 문제

### MPS (Metal Performance Shaders) 오류

#### 증상
```
RuntimeError: MPS backend out of memory
```

#### 해결 방법
```bash
# 배치 크기 줄이기
BATCH_SIZE = 16  # 32 → 16

# 또는 Fallback 활성화
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### MPS에서 특정 연산 미지원

#### 증상
```
NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device
```

#### 해결 방법
```bash
# 환경 변수 설정 (터미널에서)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 또는 Python 코드에서
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### CUDA Out of Memory (GPU 서버 사용 시)

#### 해결 방법
```python
# 1. 배치 크기 줄이기
BATCH_SIZE = 8

# 2. Gradient Accumulation 사용
GRAD_ACCUM_STEPS = 4  # 실제 배치 = 8 × 4 = 32

for step in range(steps):
    loss = model(batch)
    loss = loss / GRAD_ACCUM_STEPS
    loss.backward()

    if (step + 1) % GRAD_ACCUM_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 메모리 정리
torch.cuda.empty_cache()
```

---

## 토크나이저 문제

### 한국어가 글자 단위로 쪼개짐

#### 증상
```
"안녕하세요" → ['안', '녕', '하', '세', '요']
```

#### 원인
vocab_size가 너무 작거나 학습 데이터에 한국어가 적음

#### 해결 방법
```python
# vocab_size 늘리기
VOCAB_SIZE = 24000  # 16000 → 24000

# 또는 한국어 데이터 더 추가
```

### [UNK] 토큰이 많이 나옴

#### 원인
학습 데이터에 없는 단어/표현이 입력됨

#### 해결 방법
1. 학습 데이터에 다양한 표현 추가
2. 테스트 텍스트가 학습 도메인과 맞는지 확인
3. vocab_size 늘리기

### 토크나이저 로드 실패

#### 증상
```
FileNotFoundError: data/tokenizer.json
```

#### 해결 방법
```bash
# 토크나이저 학습 먼저 실행
python 02_train_tokenizer.py
```

---

## 학습 관련 문제

### Loss가 줄어들지 않음

#### 체크리스트

1. **학습률 확인**
```python
# 너무 높으면 발산, 너무 낮으면 수렴 안 됨
LEARNING_RATE = 3e-4  # 권장: 1e-4 ~ 5e-4
```

2. **데이터 확인**
```python
# 데이터가 올바르게 로드되는지 확인
print(f"데이터 크기: {len(train_data):,} 토큰")
print(f"첫 100개 토큰: {train_data[:100]}")
```

3. **모델 확인**
```python
# 파라미터 수 확인
params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {params:,}")
```

### Loss가 NaN 또는 Inf

#### 원인
- 학습률이 너무 높음
- Gradient Explosion

#### 해결 방법
```python
# 1. 학습률 낮추기
LEARNING_RATE = 1e-4

# 2. Gradient Clipping 적용
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 데이터에 이상한 값 있는지 확인
```

### 학습이 너무 느림

#### 최적화 방법
```python
# 1. Mixed Precision 사용 (GPU)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits, loss = model(x, targets=y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. DataLoader num_workers 늘리기
dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

# 3. 불필요한 연산 제거
with torch.no_grad():  # 추론 시
    ...
```

### 체크포인트 로드 실패

#### 증상
```
RuntimeError: Error(s) in loading state_dict for GPT
```

#### 원인
저장할 때와 로드할 때 모델 구조가 다름

#### 해결 방법
```python
# 하이퍼파라미터가 동일한지 확인
# 저장 시:
# N_LAYER=6, N_HEAD=6, N_EMBD=384

# 로드 시도 동일해야 함:
model = GPT(
    vocab_size=24000,
    block_size=256,
    n_layer=6,      # 동일해야 함
    n_head=6,       # 동일해야 함
    n_embd=384,     # 동일해야 함
    dropout=0.1
)
```

---

## 생성 관련 문제

### 생성된 텍스트가 반복됨

#### 증상
```
"연락처 연락처 연락처 연락처..."
```

#### 해결 방법
```python
# 1. Temperature 높이기
generate_text(prompt, temperature=1.0)  # 0.7 → 1.0

# 2. Top-K 늘리기
generate_text(prompt, top_k=80)  # 50 → 80

# 3. Repetition Penalty 구현 (고급)
def apply_repetition_penalty(logits, past_tokens, penalty=1.2):
    for token in set(past_tokens):
        logits[token] /= penalty
    return logits
```

### 생성된 텍스트가 엉뚱함

#### 해결 방법
```python
# 1. Temperature 낮추기
generate_text(prompt, temperature=0.7)

# 2. Top-K 줄이기
generate_text(prompt, top_k=30)

# 3. 프롬프트 형식 확인 (학습 형식과 동일해야 함)
prompt = """[QUESTION]
질문 내용
[/QUESTION]

[ANSWER]
요약:
-"""
```

### [ANSWER] 태그가 닫히지 않음

#### 원인
max_new_tokens가 부족

#### 해결 방법
```python
generate_text(prompt, max_new_tokens=600)  # 400 → 600
```

### 생성이 너무 느림

#### 최적화 방법
```python
# 1. eval() 모드 확인
model.eval()

# 2. torch.no_grad() 사용
with torch.no_grad():
    output = model.generate(...)

# 3. KV Cache 구현 (고급)
# 이전 key, value를 캐시하여 재계산 방지
```

---

## 데이터 관련 문제

### 학습 데이터가 부족함

#### 증상
- 과적합 (train loss ↓, val loss ↑)
- 생성 품질 저하

#### 해결 방법
```python
# 1. 데이터 증강
# - 같은 내용 다른 표현으로 변형
# - 동의어 치환
# - 문장 순서 섞기 (주의해서)

# 2. Dropout 높이기
DROPOUT = 0.2  # 0.1 → 0.2

# 3. 정규화 강화
WEIGHT_DECAY = 0.1  # 0.01 → 0.1
```

### 특정 형식만 잘 생성됨

#### 원인
학습 데이터의 편향

#### 해결 방법
```python
# 학습 데이터 균형 맞추기
# - 다양한 유형의 연락처 정보 포함
# - 다양한 질문 형식 포함
# - 다양한 답변 형식 포함
```

---

## 성능 최적화 팁

### 1. 배치 크기 최적화

```python
# MPS 메모리에 맞게 조절
# M4 기준: 16~32 권장

BATCH_SIZE = 32  # 메모리 여유 있으면
BATCH_SIZE = 16  # 메모리 부족하면
```

### 2. 학습률 스케줄링

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

for step in range(total_steps):
    # ... 학습 ...
    scheduler.step()
```

### 3. Early Stopping

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(max_epochs):
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 4. 효율적인 데이터 로딩

```python
import numpy as np
import torch

# 메모리 매핑으로 대용량 데이터 처리
data = np.memmap("data/train.bin", dtype=np.uint16, mode='r')

# 필요한 부분만 로드
batch = torch.from_numpy(data[start:end].astype(np.int64))
```

---

## 디버깅 팁

### 1. 텐서 형태 확인

```python
def debug_shapes(model, x):
    print(f"입력: {x.shape}")

    # 임베딩 후
    tok_emb = model.tok_emb(x)
    print(f"토큰 임베딩: {tok_emb.shape}")

    # 블록 통과 후
    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"블록 {i}: {x.shape}")
```

### 2. Gradient 확인

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
```

### 3. Attention 시각화

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    """Attention 가중치 시각화"""
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights, cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.title("Attention Weights")
    plt.savefig("attention.png")
```

---

## 자주 묻는 질문 (FAQ)

### Q: 얼마나 학습해야 하나요?

**A**: 데이터 크기에 따라 다릅니다.
- 1M 토큰: 5~10 epoch
- 10M 토큰: 2~3 epoch
- 100M+ 토큰: 1 epoch도 충분할 수 있음

Loss가 수렴하면 학습을 멈추세요.

### Q: GPU 없이도 학습 가능한가요?

**A**: 가능하지만 매우 느립니다.
- CPU: 학습에 수일~수주 소요
- MPS (Mac): 수시간~수일
- CUDA (Nvidia GPU): 수분~수시간

작은 모델로 시작하여 테스트하세요.

### Q: 더 큰 모델이 항상 좋은가요?

**A**: 아닙니다. 데이터가 적으면 큰 모델은 과적합됩니다.
- 데이터 < 1M 토큰: 작은 모델 권장
- 데이터 > 10M 토큰: 더 큰 모델 시도 가능

### Q: Temperature를 0으로 설정하면?

**A**: Greedy Decoding이 됩니다.
- 항상 가장 확률 높은 토큰 선택
- 결정적 출력 (같은 입력 → 같은 출력)
- 다양성 없음

### Q: 모델을 어떻게 배포하나요?

**A**: 여러 방법이 있습니다.
```python
# 1. 직접 로드
model = load_checkpoint("ckpt.pt")

# 2. ONNX 변환
torch.onnx.export(model, dummy_input, "model.onnx")

# 3. TorchScript
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

---

## 참고 자료

### 공식 문서
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)

### 논문
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 원본 논문
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 논문

### 구현 참고
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy의 미니 GPT 구현
- [minGPT](https://github.com/karpathy/minGPT) - 더 간단한 GPT 구현

---

## 다음 단계

축하합니다! 🎉 CAI 학습 문서를 모두 완료했습니다.

### 추천 학습 순서

1. [00-overview.md](00-overview.md) - 프로젝트 개요
2. [01-environment-setup.md](01-environment-setup.md) - 환경 설정
3. [02-project-structure.md](02-project-structure.md) - 폴더 구조
4. [03-data-preparation.md](03-data-preparation.md) - 데이터 준비
5. [04-tokenizer.md](04-tokenizer.md) - 토크나이저 학습
6. [05-model-architecture.md](05-model-architecture.md) - 모델 구조
7. [06-training.md](06-training.md) - 모델 학습
8. [07-generation.md](07-generation.md) - 텍스트 생성
9. [08-concepts.md](08-concepts.md) - 핵심 개념
10. [09-tips.md](09-tips.md) - 팁과 트러블슈팅 (현재 문서)

### 실습 순서

```bash
# 1. 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 2. 데이터 준비
uv run python scripts/prepare_samples.py

# 3. 토크나이저 학습
uv run python scripts/train_tokenizer.py

# 4. 바이너리 데이터셋 생성
uv run python scripts/build_bin_dataset.py

# 5. 모델 학습 (MPS fallback 필요)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_gpt.py

# 6. 텍스트 생성
uv run python scripts/generate.py
```

이제 여러분만의 Contact AI를 만들어보세요!
