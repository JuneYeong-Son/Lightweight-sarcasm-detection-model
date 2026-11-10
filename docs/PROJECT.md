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

### 학습 전후 비교 ([baseline_eval.py](../scripts/baseline_eval.py))

학습 전 모델과 학습 후 모델을 같은 데이터, 같은 지표로 평가했다. 상세 결과는 [baseline_eval.json](../results/baseline_eval.json)에 있다.

**비교 대상**
- **다수 클래스 기준선:** 모든 입력을 Sarcasm으로 예측한다. 데이터에 비꼼이 더 많아서(validation 60.6%, test 55.0%) 아무것도 배우지 않아도 이 정도 점수가 나온다.
- **학습 전:** klue/roberta-base 원본. 비꼼 분류용 출력층이 원본에 없어서 무작위로 초기화된다. 초기화 seed 3개(0, 1, 2)로 평가해 평균(±표준편차)을 냈다.
- **학습 후:** KoCoSa로 fine-tuning한 모델 (`model/`, 2 epoch 체크포인트).

**지표** (논문 Table 8과 같은 형식, 단위 %)
- BA: Balanced Accuracy (두 클래스 recall의 평균)
- M-F1 / W-F1: Macro-F1 / Weighted-F1
- Precision-S / Recall-S: Sarcasm 클래스의 정밀도 / 재현율
- Precision-N / Recall-N: Non-Sarcasm 클래스의 정밀도 / 재현율
- Acc, F1(S): 일반 정확도, Sarcasm 기준 이진 F1 (학습 코드에서 쓰던 지표)

**Test 셋 (1037개: Sarcasm 570, Non-Sarcasm 467)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) |
|---|---|---|---|---|---|---|---|---|---|
| 다수 클래스 (전부 Sarcasm) | 50.0 | 35.5 | 39.0 | 55.0 | 100.0 | 0.0 | 0.0 | 55.0 | 70.9 |
| 학습 전 (seed 3개 평균) | 50.0 (±0.0) | 32.5 (±2.1) | 31.6 (±5.2) | 18.3 (±25.9) | 33.3 (±47.1) | 30.0 (±21.2) | 66.7 (±47.1) | 48.3 (±4.7) | 23.6 (±33.4) |
| **학습 후 (fine-tuned)** | **72.8** | **72.5** | **73.4** | 69.8 | 96.3 | 91.6 | 49.3 | **75.1** | **81.0** |

**Validation 셋 (1321개: Sarcasm 801, Non-Sarcasm 520)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) |
|---|---|---|---|---|---|---|---|---|---|
| 다수 클래스 (전부 Sarcasm) | 50.0 | 37.7 | 45.8 | 60.6 | 100.0 | 0.0 | 0.0 | 60.6 | 75.5 |
| 학습 전 (seed 3개 평균) | 50.0 (±0.0) | 31.4 (±4.5) | 30.1 (±11.1) | 20.2 (±28.6) | 33.3 (±47.1) | 26.2 (±18.6) | 66.7 (±47.1) | 46.5 (±10.0) | 25.2 (±35.6) |
| **학습 후 (fine-tuned)** | **70.6** | **71.3** | **73.7** | 73.0 | 95.4 | 86.5 | 45.8 | **75.9** | **82.7** |

**해석**
- **학습 전 모델은 비꼼을 구분하지 못한다.** seed 0, 1은 모든 입력을 Non-Sarcasm으로, seed 2는 모두 Sarcasm으로 예측했다. 무작위 출력층은 입력과 상관없이 한쪽 클래스만 고르기 때문에 BA가 정확히 50.0(찍기 수준)이다.
- **학습 후 test BA는 50.0에서 72.8로 22.8%p 올랐다.** 정확도는 다수 클래스 기준선(55.0)보다 20.1%p 높다.
- **F1(S)만 보면 학습 효과가 작아 보인다.** 전부 Sarcasm이라고만 답해도 F1(S)가 70.9(test)나 나오기 때문이다. 학습 효과는 BA, M-F1처럼 클래스 불균형을 보정한 지표로 보는 것이 정확하다.
- **모델이 Sarcasm 쪽으로 치우쳐 있다.** Recall-S는 96.3인데 Recall-N은 49.3이다. 비꼼이 아닌 대화의 절반가량을 비꼼으로 잘못 판단한다. 논문의 KLUE-RoBERTa-base도 같은 경향을 보인다(Recall-S 93.4, Recall-N 54.7).

### 논문(KoCoSa Table 8)과 비교 — Test 셋

논문 값은 arXiv:2402.14428 Table 8에서 가져왔다. 논문에는 Acc와 F1(S)가 없다.

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N |
|---|---|---|---|---|---|---|---|
| GPT-3.5 (zero-shot) — 논문 | 53.5 | 43.4 | 40.6 | 71.2 | 20.5 | 40.2 | 86.5 |
| GPT-4 (zero-shot) — 논문 | 73.2 | 71.7 | 72.6 | 83.3 | 68.8 | 60.5 | 77.6 |
| GPT-4 (4-shot) — 논문 | 75.0 | 75.1 | 76.6 | 80.5 | 82.3 | 70.2 | 67.7 |
| KLUE-RoBERTa-base (fine-tuned) — 논문 | 74.0 (±0.2) | 74.1 (±0.6) | 74.7 (±0.2) | 71.5 (±0.2) | 93.4 (±0.5) | 87.2 (±0.7) | 54.7 (±0.7) |
| KLUE-RoBERTa-large (fine-tuned) — 논문 | 74.9 (±0.3) | 75.1 (±1.0) | 75.5 (±0.3) | 74.6 (±0.3) | 85.0 (±0.6) | 80.0 (±0.6) | 64.8 (±0.6) |
| Human — 논문 | 80.2 | 80.1 | 80.3 | 83.0 | 80.5 | 77.1 | 79.9 |
| **본 프로젝트: 학습 전 (seed 3개 평균)** | 50.0 (±0.0) | 32.5 (±2.1) | 31.6 (±5.2) | 18.3 (±25.9) | 33.3 (±47.1) | 30.0 (±21.2) | 66.7 (±47.1) |
| **본 프로젝트: 학습 후 (KLUE-RoBERTa-base)** | **72.8** | **72.5** | **73.4** | 69.8 | 96.3 | 91.6 | 49.3 |

- 같은 KLUE-RoBERTa-base 기준으로 논문보다 BA 1.2%p, M-F1 1.6%p, W-F1 1.3%p 낮다. 비슷한 수준으로 재현됐다.
- 차이는 Non-Sarcasm 재현율(49.3 vs 54.7)에서 가장 크다. Sarcasm 쪽으로 더 치우쳐 있다.
- 논문은 여러 번 학습한 평균(±)이고, 본 프로젝트는 1번 학습한 결과(seed 42)라서 직접 비교에는 한계가 있다. 논문의 학습 하이퍼파라미터와 분할 방식(약 8:1:1)이 Hugging Face 데이터셋 분할(10466 / 1321 / 1037)과 정확히 같은지도 확인하지 못했다.

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

### PC 측정 결과 (validation 1321개, 스레드 4개)

| 항목 | PyTorch | ONNX fp32 |
|---|---|---|
| 파일 크기 (MB) | 422.0 | 422.2 |
| 모델 로드 (초) | 1.82 | 0.76 |
| 지연시간 평균 / 중앙값 / p90 (ms, batch 1) | 98.1 / 95.5 / 133.4 | 69.7 / 64.6 / 100.4 |
| 모델 로드로 늘어난 메모리 (MB) | 536 | 450 |
| 프로세스 최대 메모리 (MB) | 1169 | 1061 |
| 정확도 / F1(S) | 0.7585 / 0.8273 | 0.7585 / 0.8273 |
| 평가 처리량 (개/초, batch 8) | 10.07 | 12.11 |

- 두 모델의 예측은 100% 일치했고, logit 최대 차이는 4.4e-6이다. 변환으로 인한 정확도 손실은 없다.
- ONNX는 지연시간이 약 29% 짧고, 로드 시간은 절반 이하다.
- 파일 크기는 같다. 크기를 줄이는 것은 3·4단계(pruning, 양자화)의 몫이다.
- 상세 결과: [results/benchmark/](../results/benchmark/)

### 모바일 측정 결과

(Android 측정 후 추가)

## 브랜치

| 브랜치 | 내용 |
|---|---|
| `main` | 학습 코드, 학습 결과, fine-tuned 모델 (모두 루트에 있음) |
| `feature/origin-onnx` | 원본 모델의 ONNX 변환과 기준 성능 측정. 폴더 구조를 `model/`, `onnx/`, `scripts/`, `results/`, `docs/`로 정리함 |

대용량 파일(`*.safetensors`, `*.onnx`, `*.onnx.data`)은 Git LFS로 관리한다 ([.gitattributes](../.gitattributes)).
