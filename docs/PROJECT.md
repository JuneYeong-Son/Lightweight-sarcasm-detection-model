# 한국어 비꼼(Sarcasm) 탐지 모델 경량화 프로젝트

## 목적

한국어 대화에서 비꼼을 탐지하는 분류 모델을 만들고, 이후 **pruning과 양자화로 경량화**해서 모바일에서 돌릴 수 있게 하는 것이 목표다.

| 항목 | 내용 |
|---|---|
| 기반 모델 | [klue/roberta-base](https://huggingface.co/klue/roberta-base) (12층, hidden 768, 약 1.1억 파라미터) |
| 데이터셋 | KoCoSa: Korean Context-aware Sarcasm Detection Dataset ([arXiv:2402.14428](https://arxiv.org/abs/2402.14428)), Hugging Face `YuminKim/KoCoSa` |
| 태스크 | 문장쌍 이진 분류. 입력은 `context`(대화 맥락)와 `response`(응답), 출력은 비꼼 여부 |
| 라벨 | `0` = Non-Sarcasm, `1` = Sarcasm |

## 진행 단계

1. **Full fine-tuning** (완료): [train.py](../scripts/train.py)
2. **ONNX 변환 및 기준 성능 측정** (진행 중): [export_onnx.py](../scripts/export_onnx.py), 브랜치 `feature/origin-onnx`
3. **Pruning** (예정)
4. **양자화** (예정)

2단계는 경량화를 하기 전에 원본 모델을 ONNX로 변환해서 **기준(baseline) 성능**을 확보하는 단계다. 3·4단계 결과는 모두 이 기준과 비교한다.

## 1단계: Fine-tuning

### 학습 설정 ([train.py](../scripts/train.py))

| 항목 | 값 |
|---|---|
| Epochs | 5 |
| Batch size | 16 (학습), 8 (검증) |
| Learning rate | 1e-5 (AdamW) |
| Max length | 256 |
| Gradient clipping | max_norm = 1.0 |
| Seed | 42 |
| 장치 | CPU (epoch당 약 66분) |
| 모델 저장 기준 | Validation F1이 가장 높은 epoch |

KLUE-RoBERTa는 `token_type_ids`를 쓰지 않으므로 전처리 단계에서 제거한다.

### 학습 결과 ([result.txt](../results/result.txt))

| Epoch | Train Loss | Val Accuracy | Val F1 |
|---|---|---|---|
| 1 | 0.5579 | 0.7540 | 0.8185 |
| **2** | **0.4804** | **0.7585** | **0.8273** |
| 3 | 0.4007 | 0.7623 | 0.8234 |
| 4 | 0.2964 | 0.7229 | 0.7727 |
| 5 | 0.2087 | 0.7154 | 0.7571 |

- 최고 Validation F1은 **0.8273 (2 epoch)**이고, 이 체크포인트가 저장됐다.
- 3 epoch 이후 train loss는 계속 떨어지는데 검증 성능은 하락했다. 과적합으로 보인다.

### 학습 전후 비교

자세한 내용은 [FINETUNING_COMPARISON.md](FINETUNING_COMPARISON.md)에 정리했다. 아래는 test 셋 요약이다(단위 %, 논문 Table 8 형식).

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N |
|---|---|---|---|---|---|---|---|
| 학습 전 (klue/roberta-base, seed 3개 평균) | 50.0 | 32.5 | 31.6 | 18.3 | 33.3 | 30.0 | 66.7 |
| **학습 후 (본 프로젝트)** | **72.8** | **72.5** | **73.4** | 69.8 | 96.3 | 91.6 | 49.3 |
| KLUE-RoBERTa-base (논문) | 74.0 | 74.1 | 74.7 | 71.5 | 93.4 | 87.2 | 54.7 |

- 학습 전에는 찍기 수준(BA 50.0)이고, 학습 후 BA가 22.8%p 올랐다. 논문과 비슷한 수준으로 재현됐다.
- 학습 후 모델은 Sarcasm 쪽으로 치우쳐 있다(Recall-N 49.3).

## 2단계: ONNX 변환 및 기준 성능

### 변환 ([export_onnx.py](../scripts/export_onnx.py))

- 입력: `input_ids`, `attention_mask` (batch, 시퀀스 길이 모두 가변)
- 출력: `logits` (batch × 2)
- opset 17, fp32 (양자화하지 않은 원본)
- 변환 후 PyTorch와 ONNX Runtime의 출력이 같은지 검증한다 (rtol 1e-3, atol 1e-4)
- 결과물은 `onnx/` 폴더에 저장된다 (`model.onnx` 약 440MB, 토크나이저, config)

### 측정할 항목

| 항목 | 비교 대상 |
|---|---|
| 정확도 / F1 (KoCoSa validation) | PyTorch 원본과 ONNX fp32 |
| 추론 지연시간 (평균, p90) | PyTorch 원본과 ONNX fp32 |
| 메모리 사용량 | PyTorch 원본과 ONNX fp32 |
| 모델 파일 크기 | PyTorch 원본과 ONNX fp32 |

ONNX fp32는 원본과 같은 연산이므로 정확도는 거의 같아야 한다. 실제 차이는 속도, 메모리, 배포 편의성에서 난다. 측정 스크립트는 [benchmark.py](../scripts/benchmark.py)이고, Android 측정 방법은 [MOBILE_BENCHMARK.md](MOBILE_BENCHMARK.md)에 정리했다.

### PC 측정 결과 (변환 전후 비교)

자세한 내용은 [ONNX_COMPARISON.md](ONNX_COMPARISON.md)에 정리했다. 아래는 test 셋, 스레드 4개 기준 요약이다.

| 항목 | 변환 전 (PyTorch) | 변환 후 (ONNX fp32) |
|---|---|---|
| BA / M-F1 (%) | 72.8 / 72.5 | 72.8 / 72.5 |
| 예측 일치율 | | 100% |
| 지연시간 평균 (ms, batch 1) | 118.9 | 78.4 (1.52배 빠름) |
| 모델 로드 (초) | 2.65 | 1.02 |
| 프로세스 최대 메모리 (MB) | 1,170 | 1,063 |
| 파일 크기 (MB) | 422.0 | 422.2 |

- 변환으로 인한 정확도 손실은 없다(test, validation 모두 모든 지표가 같고 예측 100% 일치).
- ONNX는 지연시간이 약 34% 짧다(스레드 1개에서는 20%).
- 파일 크기는 같다. 크기를 줄이는 것은 3·4단계(pruning, 양자화)의 몫이다.

### 모바일 측정 결과

(Android 측정 후 추가)

## 브랜치

| 브랜치 | 내용 |
|---|---|
| `main` | 학습 코드, 학습 결과, fine-tuned 모델 (모두 루트에 있음) |
| `feature/origin-onnx` | 원본 모델의 ONNX 변환과 기준 성능 측정. 폴더 구조를 `model/`, `onnx/`, `scripts/`, `results/`, `docs/`로 정리함 |

대용량 파일(`*.safetensors`, `*.onnx`, `*.onnx.data`)은 Git LFS로 관리한다 ([.gitattributes](../.gitattributes)).
