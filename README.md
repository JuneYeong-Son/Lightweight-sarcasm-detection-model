# 프루닝(0.6) + Dynamic INT8 비꼼 탐지 모델

KoCoSa 비꼼 탐지 모델(klue/roberta-base 파인튜닝)에 **비구조적 프루닝 60% + 회복학습**을 적용하고,
ONNX로 변환한 뒤 **ORTQuantizer `arm64(is_static=False)` 동적 INT8 양자화**를 적용한 결과물이다.

## 결과물

```
onnx-pruned-int8/   최종 모델 (model.onnx 106 MB + config, tokenizer)
results/            FP32 vs INT8 평가 결과, benchmark/ (①·②·③ 비교 측정)
docs/               VARIANT_COMPARISON.md (경량화 변형 5종 비교)
pipeline/           중간 과정 (아래 참고)
```

다른 경량화 변형(① ONNX fp32, ② Dynamic INT8)과의 비교는 [docs/VARIANT_COMPARISON.md](docs/VARIANT_COMPARISON.md)에 있다.
아래 표는 프루닝 모델의 FP32 ONNX와 INT8을 비교한 것이다(양자화 자체의 영향).

| | 프루닝 FP32 ONNX | 프루닝 + DynINT8 |
|---|---|---|
| 파일 크기 | 422.2 MB | **106.3 MB** |
| 지연시간 (batch 1, PC 4스레드, 중앙값) | 73 ms | **40 ms** |
| Validation Acc / F1(S) | 0.7623 / 0.8265 | 0.7570 / 0.8226 |
| Test Acc / F1(S) | 0.7599 / 0.8126 | 0.7599 / 0.8129 |
| FP32와 예측 일치율 | – | 98.4% (val) / 98.7% (test) |

## 중간 과정 (`pipeline/`)

| 단계 | 위치 | 내용 |
|---|---|---|
| 1. 파인튜닝 | `pipeline/1_finetuned_model/` | 원본 파인튜닝 모델, 토크나이저, 학습 코드(`코드.py`)와 로그(`결과.txt`) |
| 2. 프루닝 | `pipeline/2_pruning/` | `prune_recover.py`, 0.6 프루닝 모델(`models/`), 비율별 결과 |
| 3. FP32 ONNX 변환 | `pipeline/scripts/export_onnx.py` → `pipeline/3_onnx_fp32/` | Optimum export |
| 4. 동적 INT8 양자화 | `pipeline/scripts/quantize_onnx.py` → `onnx-pruned-int8/` | ORTQuantizer + arm64(is_static=False) |
| 평가 | `pipeline/scripts/evaluate_onnx.py` → `results/` | KoCoSa validation/test |

## 다시 만들기

Optimum은 transformers 4.x가 필요하므로 별도 가상환경에서 3·4단계를 실행한다.

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r pipeline/requirements-quant.txt
.venv/Scripts/python pipeline/scripts/export_onnx.py     # 3단계
.venv/Scripts/python pipeline/scripts/quantize_onnx.py   # 4단계
python pipeline/scripts/evaluate_onnx.py --split test    # 평가 (기본 Python, datasets 필요)
```
