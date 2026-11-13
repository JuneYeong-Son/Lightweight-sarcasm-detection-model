# 경량화 변형 5종 비교

fine-tuning한 비꼼 탐지 모델(klue/roberta-base + KoCoSa)을 다섯 가지 방법으로 경량화한 결과를 한 표로 비교한다. 기준은 **① ONNX fp32**이다.

## 비교 대상

| # | 변형 | Pruning | 양자화 | 모델 위치 | 상태 |
|---|---|---|---|---|---|
| ① | ONNX fp32 (기준) | 없음 | 없음 | `onnx/` | 완료 |
| ② | Dynamic INT8 | 없음 | 동적 INT8 (학습 없음) | `onnx-int8/` | 완료 |
| ③ | Pruning + Dynamic INT8 | 있음 | 동적 INT8 (학습 없음) | | 예정 |
| ④ | Pruning + QAT | 있음 | QAT (재학습, 정적 INT8) | | 예정 |
| ⑤ | QAT | 없음 | QAT (재학습, 정적 INT8) | | 예정 |

- ③과 ④는 **같은 pruning 모델**에서 출발해야 양자화 방법만의 차이를 볼 수 있다.
- ②·③(동적)과 ④·⑤(QAT)는 양자화 방식이 다르다. 동적 양자화는 활성값 범위를 추론할 때마다 계산하고, QAT는 학습으로 정한 범위를 모델에 고정한다(정적). 따라서 이 비교에는 "동적 대 정적" 차이도 함께 들어 있다.

## 요약 (PC, 스레드 4)

정확도는 test 셋(1,037개) 기준이고 단위는 %다. 변화는 ① 대비 값이다.

| 항목 | ① ONNX fp32 | ② Dynamic INT8 | ③ Pruning + DynINT8 | ④ Pruning + QAT | ⑤ QAT |
|---|---|---|---|---|---|
| 파일 크기 (MB) | 422.2 | **106.2** (−75%) | | | |
| BA | 72.8 | 72.5 (−0.3) | | | |
| M-F1 | 72.5 | 72.2 (−0.3) | | | |
| F1(S) | 81.0 | 80.9 (−0.1) | | | |
| 예측 일치율 (① 대비) | 100 | 99.2 | | | |
| 지연시간 평균 (ms, batch 1) | 53.0 | **34.9** (1.52배) | | | |
| 지연시간 평균 (ms, 스레드 1) | 154.5 | **90.0** (1.72배) | | | |
| 모델 로드로 늘어난 메모리 (MB) | 450 | **146** (−68%) | | | |
| 프로세스 최대 메모리 (MB) | 1,067 | 807 (−24%) | | | |
| 모델 로드 (초) | 0.64 | 0.25 | | | |
| 평가 처리량 (개/초, batch 8) | 15.4 | 22.4 (+46%) | | | |

## 정확도 상세

논문(arXiv:2402.14428) Table 8 형식이고 단위는 %다.

**Test 셋 (1,037개)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) | 일치율 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① ONNX fp32 | 72.8 | 72.5 | 73.4 | 69.8 | 96.3 | 91.6 | 49.3 | 75.1 | 81.0 | 100 |
| ② Dynamic INT8 | 72.5 | 72.2 | 73.1 | 69.6 | 96.5 | 91.9 | 48.6 | 74.9 | 80.9 | 99.2 |
| ③ Pruning + DynINT8 | | | | | | | | | | |
| ④ Pruning + QAT | | | | | | | | | | |
| ⑤ QAT | | | | | | | | | | |
| *참고: KLUE-RoBERTa-base (논문)* | *74.0* | *74.1* | *74.7* | *71.5* | *93.4* | *87.2* | *54.7* | | | |

**Validation 셋 (1,321개)**

| Model | BA | M-F1 | W-F1 | Precision-S | Recall-S | Precision-N | Recall-N | Acc | F1(S) | 일치율 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① ONNX fp32 | 70.6 | 71.3 | 73.7 | 73.0 | 95.4 | 86.5 | 45.8 | 75.9 | 82.7 | 100 |
| ② Dynamic INT8 | 70.7 | 71.4 | 73.9 | 72.9 | 96.5 | 89.3 | 44.8 | 76.2 | 83.1 | 98.8 |
| ③ Pruning + DynINT8 | | | | | | | | | | |
| ④ Pruning + QAT | | | | | | | | | | |
| ⑤ QAT | | | | | | | | | | |

일치율은 ①과 예측이 같은 샘플의 비율이다. ②에서 예측이 달라진 샘플은 test 8개, validation 16개다.

## 속도와 메모리 상세

**지연시간 (batch 1, 단위 ms)**

| 스레드 | 모델 | 평균 | 중앙값 | p90 | 최소 | 최대 |
|---|---|---|---|---|---|---|
| 4 | ① ONNX fp32 | 53.0 | 50.7 | 70.4 | 30.8 | 92.5 |
| 4 | ② Dynamic INT8 | 34.9 | 33.7 | 47.2 | 16.8 | 60.6 |
| 4 | ③ Pruning + DynINT8 | | | | | |
| 4 | ④ Pruning + QAT | | | | | |
| 4 | ⑤ QAT | | | | | |
| 1 | ① ONNX fp32 | 154.5 | 146.3 | 209.2 | 61.7 | 255.2 |
| 1 | ② Dynamic INT8 | 90.0 | 84.6 | 125.3 | 33.4 | 153.0 |
| 1 | ③ Pruning + DynINT8 | | | | | |
| 1 | ④ Pruning + QAT | | | | | |
| 1 | ⑤ QAT | | | | | |

스레드 4는 test 셋, 스레드 1은 validation 셋의 앞쪽 32개 문장(평균 약 103 토큰)으로 쟀다. 5번 예열한 뒤 50번 측정했다.

**메모리와 크기**

| 모델 | 파일 크기 (MB) | 로드로 늘어난 메모리 (MB) | 프로세스 최대 메모리 (MB) | 모델 로드 (초) |
|---|---|---|---|---|
| ① ONNX fp32 | 422.2 | 450 | 1,067 | 0.64 |
| ② Dynamic INT8 | 106.2 | 146 | 807 | 0.25 |
| ③ Pruning + DynINT8 | | | | |
| ④ Pruning + QAT | | | | |
| ⑤ QAT | | | | |

모바일(Android) 결과는 측정한 뒤 같은 형식으로 추가한다([MOBILE_BENCHMARK.md](MOBILE_BENCHMARK.md)).

## 측정 조건

| 항목 | 값 |
|---|---|
| 장치 | PC (Windows 10, CPU 12 논리 코어), Python 3.13.5 |
| 런타임 | ONNX Runtime 1.31.1, CPUExecutionProvider, ORT_ENABLE_ALL |
| 입력 | `context` + `response` 문장쌍, 최대 256 토큰 |
| 정확도 평가 | 데이터 전체, batch 8 |
| 스크립트 | [benchmark.py](../scripts/benchmark.py) (모델마다 별도 프로세스, 모델 사이 10초 대기) |

**같은 실행 안에서 비교한다.** 같은 PC에서도 실행할 때마다 지연시간의 절대값이 20% 안팎으로 달라진다. 예를 들어 ① ONNX fp32는 이전 측정([ONNX_COMPARISON.md](ONNX_COMPARISON.md))에서 78.4 ms였고 이번에는 53.0 ms였다. 그래서 이 표의 ①과 ②는 한 번의 실행에서 함께 잰 값을 썼다. ③~⑤를 추가할 때도 ①과 같은 실행에서 함께 재야 공정하다.

## 결과 파일과 재현 방법

| 파일 | 내용 |
|---|---|
| [20261113_190050_pc_int8_t4_test.json](../results/benchmark/20261113_190050_pc_int8_t4_test.json) | ①·②, test 셋, 스레드 4 |
| [20261113_190347_pc_int8_t4_validation.json](../results/benchmark/20261113_190347_pc_int8_t4_validation.json) | ①·②, validation 셋, 스레드 4 |
| [20261113_190427_pc_int8_t1_latency.json](../results/benchmark/20261113_190427_pc_int8_t1_latency.json) | ①·②, 속도만, 스레드 1 |

```powershell
python scripts\quantize_onnx.py                                   # ② 만들기
python scripts\benchmark.py --backends onnx onnx-int8 --split test --threads 4
python scripts\benchmark.py --backends onnx onnx-int8 --split validation --threads 4
python scripts\benchmark.py --backends onnx onnx-int8 --no-eval --threads 1
```

②를 만드는 과정은 [quantize_onnx.py](../scripts/quantize_onnx.py)를 참고한다.
