# CAI (Contact AI)

전 세계의 개인, 회사, 각종 서비스 연락처를 제공하는 나만의 LLM을 처음부터(from scratch) 만드는 프로젝트입니다.

## 프로젝트 목표

- **100% 직접 구현**: 파인튜닝이 아닌, 토크나이저부터 GPT 모델까지 처음부터 학습
- **요약/정리형 출력**: 체크리스트, 연락처, 상세 설명을 구조화된 형식으로 생성
- **Mac M4 최적화**: MPS(Metal Performance Shaders) GPU 가속 지원

## 빠른 시작

```bash
# 1. 의존성 설치 (uv 사용)
uv add torch tokenizers tqdm numpy

# 2. 순차 실행
uv run python scripts/prepare_samples.py      # 데이터 전처리
uv run python scripts/train_tokenizer.py      # 토크나이저 학습
uv run python scripts/build_bin_dataset.py    # 바이너리 데이터셋 생성
uv run python scripts/train_gpt.py         # GPT 모델 학습
uv run python scripts/generate.py             # 텍스트 생성
```

## 프로젝트 구조

```
cai/
  data/                  # 데이터 디렉토리
  scripts/               # 실행 스크립트
    prepare_samples.py   # 데이터 전처리
    train_tokenizer.py   # 토크나이저 학습
    build_bin_dataset.py # 바이너리 변환
    train_gpt.py         # GPT 학습
    generate.py          # 텍스트 생성
  checkpoints/           # 모델 체크포인트
  pyproject.toml         # uv 프로젝트 설정
```

## 문서 목록

상세한 학습 자료는 `docs/` 폴더에서 확인하세요.

| 문서 | 내용 |
|------|------|
| [00-overview.md](docs/00-overview.md) | 프로젝트 개요, 목표, 출력 형식 예시 |
| [01-environment-setup.md](docs/01-environment-setup.md) | Python 환경, PyTorch, MPS 설정 |
| [02-project-structure.md](docs/02-project-structure.md) | 폴더 구조, 파일 설명, 실행 순서 |
| [03-data-preparation.md](docs/03-data-preparation.md) | 데이터 수집, 전처리, 학습 형식 |
| [04-tokenizer.md](docs/04-tokenizer.md) | BPE 토크나이저 학습 |
| [05-model-architecture.md](docs/05-model-architecture.md) | GPT 모델 아키텍처, 코드 |
| [06-training.md](docs/06-training.md) | 학습 루프, 체크포인트 |
| [07-generation.md](docs/07-generation.md) | 텍스트 생성, 샘플링 파라미터 |
| [08-concepts.md](docs/08-concepts.md) | 9가지 핵심 개념 설명 |
| [09-tips.md](docs/09-tips.md) | 팁과 트러블슈팅 |

## 핵심 개념

1. **Tokenizer**: 텍스트 → 정수 토큰 변환
2. **Embedding**: 토큰 ID → 고차원 벡터
3. **Self-Attention**: 문맥 내 단어 간 관계 학습
4. **Next-token Prediction**: GPT의 유일한 학습 목표
5. **데이터 포맷 = 모델 능력**: 요약형 데이터로 학습하면 요약형으로 출력

## 권장 하이퍼파라미터 (M4 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| vocab_size | 24,000 | 토크나이저 어휘 크기 |
| block_size | 256 | 컨텍스트 길이 |
| n_layer | 6 | Transformer 블록 수 |
| n_head | 6 | Attention Head 수 |
| n_embd | 384 | 임베딩 차원 |
| batch_size | 16 | 배치 크기 |

## 출력 예시

```
[ANSWER]
요약:
- 주요 도시 한국 대사관/영사관 위치 및 연락처 정리
- 업무시간: 평일 09:00-17:00

체크리스트:
- 해야 할 일:
  - (1) 방문 전 전화 예약
  - (2) 필요 서류 사전 확인
- 준비물:
  - (1) 여권 원본
  - (2) 신청서

연락처(공공정보):
- 대한민국 대사관
  - TEL: +1-202-939-5600
  - ADDR: 2450 Massachusetts Ave NW, Washington, DC
  - WEB: https://overseas.mofa.go.kr/

상세 설명:
...
[/ANSWER]
```

## 참고 자료

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) - Karpathy의 GPT 구현 튜토리얼
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - BPE 토크나이저 문서
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html) - Mac GPU 가속 문서
