CAI 인공지능을 만든 과정 설명
=====

# 개요
- CAI 프로젝트는 전 세계의 개인이나 회사, 서비스 연락처 정보 생성에 특화된 LLM을 만드는 학습 프로젝트이다.
- 활용: 개인 또는 회사, 서비스의 연락처를 알고자하는 경우 쓸 수 있다.
  예를 들면, 필리핀 마닐라에 긴급 상황이 발생하여 한국 대사관에 연락을 해야하는데 휴일이라 대사관이 근무하지 않는 경우, 영사관이나 코리안 경찰 데스크톱의 일원(개인)에게 개인 정보(개인 전화) 연락처로 연락 해야 한다. 또한 이러한 비상 상태의 연락처는 공공 정보로 대중에게 오픈된 개인 연락처 정보이다. 이러한 공공성이 있는 개인적인 연락처를 포함한 연락처 정보를 생성하는 LLM을 만드는 프로젝트이다.

# 프로젝트 생성

```bash
% mkdir cai
% cd cai
% uv venv
% uv init
```


# 외부 라이브러리 추가

```bash
uv add torch tokenizers tqdm numpy
```

| 패키지 | 용도 |
|--------|------|
| `torch` | PyTorch 딥러닝 프레임워크 |
| `tokenizers` | Hugging Face BPE 토크나이저 |
| `tqdm` | 진행률 표시 |
| `numpy` | 수치 연산 |


# MPS 설정 확인

M1 맥북에서 MPS(Metal Performance Shaders) 사용 가능 여부를 확인하기.

아래의 코드를 실행하면 된다.

```bash
# 터미널에서 직접 실행
uv run python -c "import torch; print('MPS 사용 가능:', torch.backends.mps.is_available())"
```

