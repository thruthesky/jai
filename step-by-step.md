# JAI 인공지능 만들기: 단계별 가이드

본 문서를 따라하면 자신만의 Job LLM을 처음부터 직접 구현하고 학습시킬 수 있습니다.

---

## 개요

**JAI(Job AI)**는 전 세계 구인/구직 정보 생성에 특화된 LLM을 만드는 학습 프로젝트입니다.

### 활용 예시

> 해외 취업을 준비 중 → 미국 실리콘밸리 소프트웨어 엔지니어 채용 정보가 필요함 → JAI에게 질문

이런 상황에서 전 세계에 흩어져 있는 구인/구직 정보를 제공하는 LLM입니다.

---

## 1단계: 프로젝트 생성

```bash
mkdir jai
cd jai
uv init
```

`uv init` 명령어가 `pyproject.toml` 파일을 생성합니다.

---

## 2단계: 패키지 설치

```bash
uv add torch tokenizers tqdm numpy
```

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `tokenizers` | BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |

---

## 3단계: MPS 확인

Apple Silicon Mac에서 GPU 가속 사용 가능 여부를 확인합니다.

```bash
uv run python -c "import torch; print('MPS 사용 가능:', torch.backends.mps.is_available())"
```

`True`가 출력되면 GPU 가속을 사용할 수 있습니다.

---

## 4단계: 폴더 구조 생성

```bash
mkdir -p data scripts checkpoints
```

```
jai/
├── data/           # 데이터 파일
├── scripts/        # Python 스크립트
├── checkpoints/    # 모델 체크포인트
└── pyproject.toml  # 프로젝트 설정
```

---

## 5단계: 데이터 준비

`data/raw.txt`에 원본 구인/구직 데이터를 준비합니다.

```bash
uv run python scripts/prepare_samples.py
```

**결과**: `data/samples.txt` 생성

---

## 6단계: 토크나이저 학습

텍스트를 토큰으로 변환하는 BPE 토크나이저를 학습합니다.

```bash
uv run python scripts/train_tokenizer.py
```

**결과**: `data/tokenizer.json` 생성

---

## 7단계: 바이너리 데이터셋 생성

토큰화된 데이터를 학습용 바이너리 파일로 변환합니다.

```bash
uv run python scripts/build_bin_dataset.py
```

**결과**: `data/train.bin`, `data/val.bin` 생성

---

## 8단계: GPT 모델 학습

Transformer 기반 GPT 모델을 학습합니다.

```bash
uv run python scripts/train_gpt.py
```

**결과**: `checkpoints/ckpt.pt` 생성

학습 중 다음과 같은 출력이 표시됩니다:
```
step 0: train loss 10.234, val loss 10.198
step 100: train loss 6.543, val loss 6.612
...
```

Loss가 점점 줄어들면 학습이 잘 되고 있는 것입니다.

---

## 9단계: 텍스트 생성

학습된 모델로 텍스트를 생성합니다.

```bash
uv run python scripts/generate.py
```

**예시 출력**:
```
[QUESTION]
미국 실리콘밸리 소프트웨어 엔지니어 채용
[/QUESTION]

[ANSWER]
요약:
- 실리콘밸리 주요 기업 채용 정보

구인 정보:
- Google Inc.
  - 포지션: Senior Software Engineer
  - 연봉: $150,000 - $200,000
  - 위치: Mountain View, CA
[/ANSWER]
```

---

## 전체 실행 순서 요약

```bash
# 1. 프로젝트 설정
uv init
uv add torch tokenizers tqdm numpy
mkdir -p data scripts checkpoints

# 2. 순차 실행
uv run python scripts/prepare_samples.py
uv run python scripts/train_tokenizer.py
uv run python scripts/build_bin_dataset.py
uv run python scripts/train_gpt.py
uv run python scripts/generate.py
```

---

## 다음 단계

각 스크립트의 상세 구현은 `docs/` 폴더를 참조하세요:

- [03-data-preparation.md](docs/03-data-preparation.md) - 데이터 전처리
- [04-tokenizer.md](docs/04-tokenizer.md) - 토크나이저
- [06-training.md](docs/06-training.md) - GPT 학습
- [07-generation.md](docs/07-generation.md) - 텍스트 생성
