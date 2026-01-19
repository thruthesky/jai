# 팁과 트러블슈팅

## 개요

JAI 개발 중 자주 발생하는 문제와 해결 방법을 정리합니다.

---

## 환경 관련 문제

### MPS 메모리 오류

```
RuntimeError: MPS backend out of memory
```

**해결**: 배치 크기 줄이기 (16 → 8 → 4)

### MPS 연산 미지원

```
NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device
```

**해결**: 스크립트에 MPS fallback이 이미 포함되어 있으므로, 이 오류가 발생하면 스크립트 상단에 아래 코드가 있는지 확인:
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

---

## 토크나이저 문제

### 한국어가 글자 단위로 쪼개짐

**원인**: vocab_size가 너무 작거나 학습 데이터에 한국어가 적음

**해결**: vocab_size 늘리기 (24000 권장)

### [UNK] 토큰이 많이 나옴

**해결**:
1. 학습 데이터에 다양한 표현 추가
2. vocab_size 늘리기

### 토크나이저 로드 실패

```bash
# 토크나이저 학습 먼저 실행
uv run python scripts/train_tokenizer.py
```

---

## 학습 관련 문제

### Loss가 줄어들지 않음

**체크리스트**:
1. 학습률 확인 (권장: 1e-4 ~ 5e-4)
2. 데이터 로드 확인
3. 모델 파라미터 수 확인

### Loss가 NaN 또는 Inf

**해결**:
```python
# 학습률 낮추기
LEARNING_RATE = 1e-4

# Gradient Clipping 적용
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 체크포인트 로드 실패

**원인**: 저장할 때와 로드할 때 모델 구조가 다름

**해결**: 하이퍼파라미터(n_layer, n_head, n_embd)가 동일한지 확인

---

## 생성 관련 문제

### 생성된 텍스트가 반복됨

**해결**:
```python
# Temperature 높이기
generate_text(prompt, temperature=1.0)

# Top-K 늘리기
generate_text(prompt, top_k=80)
```

### 생성된 텍스트가 엉뚱함

**해결**:
```python
# Temperature 낮추기
generate_text(prompt, temperature=0.7)

# Top-K 줄이기
generate_text(prompt, top_k=30)
```

### [ANSWER] 태그가 닫히지 않음

**해결**: max_new_tokens 늘리기
```python
generate_text(prompt, max_new_tokens=600)
```

---

## 데이터 관련 문제

### 과적합 (train loss ↓, val loss ↑)

**해결**:
1. 데이터 더 추가
2. Dropout 높이기 (0.1 → 0.2)
3. 정규화 강화

---

## 자주 묻는 질문 (FAQ)

### Q: 얼마나 학습해야 하나요?

Loss가 수렴하면 학습을 멈추세요.
- 1M 토큰: 5~10 epoch
- 10M 토큰: 2~3 epoch

### Q: Temperature를 0으로 설정하면?

Greedy Decoding이 됩니다.
- 항상 가장 확률 높은 토큰 선택
- 다양성 없음

---

## 참고 자료

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy의 미니 GPT 구현
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)

---

## 실습 순서

```bash
# 1. 의존성 설치
uv add torch tokenizers tqdm numpy

# 2. 데이터 준비
uv run python scripts/prepare_samples.py

# 3. 토크나이저 학습
uv run python scripts/train_tokenizer.py

# 4. 바이너리 데이터셋 생성
uv run python scripts/build_bin_dataset.py

# 5. 모델 학습
uv run python scripts/train_gpt.py

# 6. 텍스트 생성
uv run python scripts/generate.py
```
