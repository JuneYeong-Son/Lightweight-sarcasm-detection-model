# 경량화 변형 5종 비교

> 이 문서는 `feature/prunning_x_DynInt8` 브랜치의 `docs/VARIANT_COMPARISON.md`와 같은 내용이다. ①·② 모델과 측정 스크립트(`scripts/benchmark.py`)는 그 브랜치에 있고, 이 브랜치에는 ③ 모델과 측정 결과 파일만 있다.

fine-tuning한 비꼼 탐지 모델(klue/roberta-base + KoCoSa)을 다섯 가지 방법으로 경량화한 결과를 한 표로 비교한다. 기준은 **① ONNX fp32**이다.

## 비교 대상

| # | 변형 | Pruning | 양자화 | 모델 위치 | 상태 |
|---|---|---|---|---|---|
| ① | ONNX fp32 (기준) | 없음 | 없음 | `onnx/` | 완료 |
| ② | Dynamic INT8 | 없음 | 동적 INT8 (학습 없음) | `onnx-int8/` | 완료 |
| ③ | Pruning + Dynamic INT8 | 비구조적 60% + 회복학습 | 동적 INT8 (학습 없음) | `onnx-pruned-int8/` (이 브랜치) | 완료 |
| ④ | Pruning + QAT | 있음 | QAT (재학습, 정적 INT8) | | 예정 |
| ⑤ | QAT | 없음 | QAT (재학습, 정적 INT8) | | 예정 |

- ③과 ④는 **같은 pruning 모델**에서 출발해야 양자화 방법만의 차이를 볼 수 있다. ③에 쓴 pruning 모델은 이 브랜치의 `pipeline/2_pruning/models/`(L1 비구조적 60%, 회복학습 5 epoch)이다.
- ②·③(동적)과 ④·⑤(QAT)는 양자화 방식이 다르다. 동적 양자화는 활성값 범위를 추론할 때마다 계산하고, QAT는 학습으로 정한 범위를 모델에 고정한다(정적). 따라서 이 비교에는 "동적 대 정적" 차이도 함께 들어 있다.

## 요약 (PC, 스레드 4)

정확도는 test 셋(1,037개) 기준이고 단위는 %다. 변화는 ① 대비 값이다. 속도와 메모리는 ①·②·③을 한 번의 실행에서 함께 잰 값이다(2026-11-18).

| 항목 | ① ONNX fp32 | ② Dynamic INT8 | ③ Pruning + DynINT8 | ④ Pruning + QAT | ⑤ QAT |
|---|---|---|---|---|---|
| 파일 크기 (MB) | 422.2 | **106.2** (−75%) | 106.3 (−75%) | | |
| BA | 72.8 | 72.5 (−0.3) | **73.9** (+1.1) | | |
| M-F1 | 72.5 | 72.2 (−0.3) | **73.9** (+1.4) | | |
| F1(S) | 81.0 | 80.9 (−0.1) | **81.3** (+0.3) | | |
| 예측 일치율 (① 대비) | 100 | 99.2 | 93.9 | | |
| 지연시간 평균 (ms, batch 1) | 69.3 | 42.9 (1.62배) | **37.7** (1.84배) | | |
| 지연시간 평균 (ms, 스레드 1) | 165.1 | **104.6** (1.58배) | 115.3 (1.43배) | | |
| 모델 로드로 늘어난 메모리 (MB) | 449 | 146 (−67%) | **145** (−68%) | | |
| 프로세스 최대 메모리 (MB) | 1,064 | 803 (−25%) | **730** (−31%) | | |
| 모델 로드 (초) | 0.77 | **0.32** | 0.47 | | |
| 평가 처리량 (개/초, batch 8) | 11.8 | 19.2 (+63%) | **20.2** (+72%) | | |

## ③ Pruning + DynINT8 결과 해석

- **크기와 속도는 ②와 사실상 같다.** 비구조적 pruning은 가중치의 60%를 0으로 만들지만, ONNX는 0도 일반 숫자처럼 저장하고 계산한다. 그래서 파일 크기(106.3 MB)와 연산량이 ②와 같다. 지연시간도 test 측정(스레드 4)에서는 ③이, validation 측정(스레드 4)과 스레드 1 측정에서는 ②가 조금 빨랐다. 일정한 차이가 아니라 측정 편차 범위다.
- **정확도는 ①보다 조금 높다.** test에서 BA +1.1, M-F1 +1.4, validation에서 BA +0.5, M-F1 +0.6이다. Sarcasm 재현율(Recall-S)은 내려가고 Non-Sarcasm 재현율(Recall-N)이 올라서, 비꼼 쪽으로 쏠렸던 예측이 조금 균형을 찾았다.
- **① 대비 일치율(93~94%)이 낮은 것은 양자화 때문이 아니다.** ③은 pruning 뒤 회복학습을 다시 한 다른 모델이라 ①과 예측이 달라진다. 같은 pruning 모델의 fp32 ONNX와 비교하면 INT8 양자화 후 일치율은 test 98.7%, validation 98.4%로 ②와 비슷하다([results/pruned_dynint8_test.json](../results/pruned_dynint8_test.json), [validation](../results/pruned_dynint8_validation.json)).
- **크기나 속도까지 줄이려면** 구조적 pruning(헤드·뉴런 단위로 실제 제거)이나 희소 행렬을 지원하는 런타임이 필요하다.

## 정확도 상세

논문(arXiv:2402.14428) Table 8 형식이고 단위는 %다.

**Test 셋 (1,037개)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) | 일치율 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① ONNX fp32 | 72.8 | 72.5 | 73.4 | 69.8 | 96.3 | 91.6 | 49.3 | 75.1 | 81.0 | 100 |
| ② Dynamic INT8 | 72.5 | 72.2 | 73.1 | 69.6 | 96.5 | 91.9 | 48.6 | 74.9 | 80.9 | 99.2 |
| ③ Pruning + DynINT8 | 73.9 | 73.9 | 74.6 | 71.1 | 94.9 | 89.5 | 52.9 | 76.0 | 81.3 | 93.9 |
| ④ Pruning + QAT | | | | | | | | | | |
| ⑤ QAT | | | | | | | | | | |
| *참고: KLUE-RoBERTa-base (논문)* | *74.0* | *74.1* | *74.7* | *71.5* | *93.4* | *87.2* | *54.7* | | | |

**Validation 셋 (1,321개)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) | 일치율 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① ONNX fp32 | 70.6 | 71.3 | 73.7 | 73.0 | 95.4 | 86.5 | 45.8 | 75.9 | 82.7 | 100 |
| ② Dynamic INT8 | 70.7 | 71.4 | 73.9 | 72.9 | 96.5 | 89.3 | 44.8 | 76.2 | 83.1 | 98.8 |
| ③ Pruning + DynINT8 | 71.1 | 71.9 | 74.1 | 73.8 | 92.9 | 81.8 | 49.2 | 75.7 | 82.3 | 93.5 |
| ④ Pruning + QAT | | | | | | | | | | |
| ⑤ QAT | | | | | | | | | | |

일치율은 ①과 예측이 같은 샘플의 비율이다. ①과 예측이 달라진 샘플은 ②가 test 8개, validation 16개이고, ③이 test 63개, validation 86개다.

## 속도와 메모리 상세

**지연시간 (batch 1, 단위 ms)**

| 스레드 | 모델 | 평균 | 중앙값 | p90 | 최소 | 최대 |
|---|---|---|---|---|---|---|
| 4 | ① ONNX fp32 | 69.3 | 63.9 | 92.7 | 34.6 | 109.9 |
| 4 | ② Dynamic INT8 | 42.9 | 42.6 | 59.1 | 22.3 | 65.5 |
| 4 | ③ Pruning + DynINT8 | 37.7 | 37.2 | 50.4 | 21.3 | 59.9 |
| 4 | ④ Pruning + QAT | | | | | |
| 4 | ⑤ QAT | | | | | |
| 1 | ① ONNX fp32 | 165.1 | 161.7 | 224.8 | 63.3 | 283.8 |
| 1 | ② Dynamic INT8 | 104.6 | 100.9 | 138.7 | 34.9 | 175.6 |
| 1 | ③ Pruning + DynINT8 | 115.3 | 111.0 | 155.9 | 42.5 | 231.1 |
| 1 | ④ Pruning + QAT | | | | | |
| 1 | ⑤ QAT | | | | | |

스레드 4는 test 셋, 스레드 1은 validation 셋의 앞쪽 32개 문장(평균 약 103 토큰)으로 쟀다. 5번 예열한 뒤 50번 측정했다.

**메모리와 크기**

| 모델 | 파일 크기 (MB) | 로드로 늘어난 메모리 (MB) | 프로세스 최대 메모리 (MB) | 모델 로드 (초) |
|---|---|---|---|---|
| ① ONNX fp32 | 422.2 | 449 | 1,064 | 0.77 |
| ② Dynamic INT8 | 106.2 | 146 | 803 | 0.32 |
| ③ Pruning + DynINT8 | 106.3 | 145 | 730 | 0.47 |
| ④ Pruning + QAT | | | | |
| ⑤ QAT | | | | |

모바일(Android) 결과는 측정한 뒤 같은 형식으로 추가한다(`MOBILE_BENCHMARK.md` (`feature/prunning_x_DynInt8` 브랜치)).

## 측정 조건

| 항목 | 값 |
|---|---|
| 장치 | PC (Windows 10, CPU 12 논리 코어), Python 3.13.5 |
| 런타임 | ONNX Runtime 1.31.1, CPUExecutionProvider, ORT_ENABLE_ALL |
| 입력 | `context` + `response` 문장쌍, 최대 256 토큰 |
| 정확도 평가 | 데이터 전체, batch 8 |
| 스크립트 | `benchmark.py` (`feature/prunning_x_DynInt8` 브랜치) (모델마다 별도 프로세스, 모델 사이 10초 대기) |

**같은 실행 안에서 비교한다.** 같은 PC에서도 실행할 때마다 지연시간의 절대값이 20% 안팎으로 달라진다. 예를 들어 ① ONNX fp32의 평균 지연시간(스레드 4)은 `ONNX_COMPARISON.md` (`feature/prunning_x_DynInt8` 브랜치) 측정에서 78.4 ms, 11/13 측정에서 53.0 ms, 11/18 측정에서 69.3 ms였다. 그래서 ③을 추가하면서 ①·②·③을 한 번의 실행에서 다시 쟀고, 이 문서의 속도·메모리 값은 모두 11/18 측정값이다. 정확도는 실행마다 같아서 ①·②의 값이 이전 측정과 같다. ④·⑤를 추가할 때도 ①과 같은 실행에서 함께 재야 공정하다.

**양자화 도구.** ②는 `onnxruntime.quantization.quantize_dynamic`(가중치 int8, per-tensor), ③은 Optimum `ORTQuantizer` + `AutoQuantizationConfig.arm64(is_static=False)`(QOperator, 활성값 u8 / 가중치 s8, per-tensor)로 만들었다. 두 방법 모두 ONNX Runtime의 같은 동적 양자화를 쓴다.

## 결과 파일과 재현 방법

| 파일 | 내용 |
|---|---|
| [20261118_172843_pc_pruned_int8_t4_test.json](../results/benchmark/20261118_172843_pc_pruned_int8_t4_test.json) | ①·②·③, test 셋, 스레드 4 |
| [20261118_173342_pc_pruned_int8_t4_validation.json](../results/benchmark/20261118_173342_pc_pruned_int8_t4_validation.json) | ①·②·③, validation 셋, 스레드 4 |
| [20261118_173454_pc_pruned_int8_t1_latency.json](../results/benchmark/20261118_173454_pc_pruned_int8_t1_latency.json) | ①·②·③, 속도만, 스레드 1 |
| `20261113_190050_pc_int8_t4_test.json` (`feature/prunning_x_DynInt8` 브랜치) | (이전) ①·②, test 셋, 스레드 4 |
| `20261113_190347_pc_int8_t4_validation.json` (`feature/prunning_x_DynInt8` 브랜치) | (이전) ①·②, validation 셋, 스레드 4 |
| `20261113_190427_pc_int8_t1_latency.json` (`feature/prunning_x_DynInt8` 브랜치) | (이전) ①·②, 속도만, 스레드 1 |

```powershell
python scripts\quantize_onnx.py                                   # ② 만들기
# ③ 모델은 feature/prunning_o_DynINT8 브랜치에서 만든다 (pipeline/scripts/export_onnx.py → quantize_onnx.py)
$P = "<feature/prunning_o_DynINT8 폴더>\onnx-pruned-int8"
python scripts\benchmark.py --backends onnx onnx-int8 onnx-pruned-int8 --onnx-pruned-int8-dir $P --split test --threads 4
python scripts\benchmark.py --backends onnx onnx-int8 onnx-pruned-int8 --onnx-pruned-int8-dir $P --split validation --threads 4
python scripts\benchmark.py --backends onnx onnx-int8 onnx-pruned-int8 --onnx-pruned-int8-dir $P --no-eval --threads 1
```

②를 만드는 과정은 `quantize_onnx.py` (`feature/prunning_x_DynInt8` 브랜치)를 참고한다.
