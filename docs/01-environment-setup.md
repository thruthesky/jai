# 환경 설정

## 개요

JAI 프로젝트는 **Macbook M4 (MPS GPU)**를 기준으로 설계되었습니다.
Apple Silicon은 PyTorch의 MPS(Metal Performance Shaders) GPU 가속을 지원합니다.

## 1. uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv
```

## 2. 필수 패키지 설치

```bash
cd cai
uv add torch tokenizers tqdm numpy
```

### 패키지 설명

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `tokenizers` | Hugging Face BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |

## 3. MPS 확인

MPS(Metal Performance Shaders)는 Apple Silicon Mac에서 GPU 가속을 사용하는 기능입니다.

```bash
# 터미널에서 직접 실행
uv run python -c "import torch; print('MPS 사용 가능:', torch.backends.mps.is_available())"
```

`True`가 출력되면 GPU 가속 사용 가능합니다.

```python
# 또는 Python 코드에서 확인
import torch

print(torch.backends.mps.is_available())  # True면 OK
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

## 다음 단계

- [02-project-structure.md](02-project-structure.md) - 프로젝트 폴더 구조
