print("===== evaluate_qat.py 실행 시작 =====", flush=True)

import csv
import gc
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from torchao.quantization import quantize_
from torchao.quantization.qat import (
    QATConfig,
    IntxFakeQuantizeConfig,
    FakeQuantizedLinear,
)

# 1. 경로 및 환경
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent
QAT_DIR = PROJECT_DIR / "qat_model"
CHECKPOINT = QAT_DIR / "best_qat_prepared.pt"
EXPORT_DIR = QAT_DIR / "qat_float_model"

MAX_LENGTH = 256
BATCH_SIZE = 8

torch.set_num_threads(4)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"QAT 체크포인트가 없습니다: {CHECKPOINT}\n"
        "먼저 train_qat.py 전체 학습을 완료하세요."
    )

# 2. 토크나이저 및 Test 데이터셋
tokenizer = AutoTokenizer.from_pretrained(
    str(BASE_DIR),
    local_files_only=True,
)

dataset = load_dataset("YuminKim/KoCoSa")
test_raw = dataset["test"]

label_map = {
    "Non-Sarcasm": 0,
    "Sarcasm": 1,
}

def preprocess(examples):
    result = tokenizer(
        examples["context"],
        examples["response"],
        truncation=True,
        max_length=MAX_LENGTH,
    )
    result.pop("token_type_ids", None)
    result["labels"] = [
        label_map[x] for x in examples["label"]
    ]
    return result

test_data = test_raw.map(
    preprocess,
    batched=True,
    remove_columns=test_raw.column_names,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer),
)

print("Test 데이터 수:", len(test_data))

# 3. 예측 및 지표 계산
def predict(model):
    model.eval()
    predictions = []
    labels = []

    with torch.inference_mode():
        for batch in test_loader:
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

            predictions.extend(preds.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return labels, predictions

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "BA": balanced_accuracy_score(y_true, y_pred) * 100,
        "M-F1": f1_score(
            y_true, y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ) * 100,
        "F1(S)": f1_score(
            y_true, y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        ) * 100,
    }

def print_metrics(name, metrics):
    print(f"\n[{name}]")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}%")

# 4. 기존 FP32 모델 평가
print("\n기존 FP32 모델 평가 중...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    str(BASE_DIR),
    local_files_only=True,
)

y_true, base_preds = predict(base_model)
base_metrics = calculate_metrics(y_true, base_preds)
print_metrics("FP32", base_metrics)

del base_model
gc.collect()

# 5. QAT 모델 복원
print("\nQAT 모델 복원 중...")

qat_model = AutoModelForSequenceClassification.from_pretrained(
    str(BASE_DIR),
    local_files_only=True,
)

activation_config = IntxFakeQuantizeConfig(
    torch.int8,
    "per_token",
    is_symmetric=False,
)

weight_config = IntxFakeQuantizeConfig(
    torch.int8,
    "per_channel",
    is_symmetric=True,
)

quantize_(
    qat_model,
    QATConfig(
        activation_config=activation_config,
        weight_config=weight_config,
        step="prepare",
    ),
    filter_fn=lambda module, name: isinstance(
        module, torch.nn.Linear
    ),
)

layer_count = sum(
    isinstance(module, FakeQuantizedLinear)
    for module in qat_model.modules()
)

print("Fake Quantized Linear 계층:", layer_count)

state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)
qat_model.load_state_dict(state, strict=True)
del state
gc.collect()

# 6. QAT Fake Quantization 모델 평가
print("\nQAT Fake Quantization 모델 평가 중...")

_, qat_preds = predict(qat_model)
qat_metrics = calculate_metrics(y_true, qat_preds)

print_metrics("QAT Fake Quant", qat_metrics)

agreement = (
    sum(a == b for a, b in zip(base_preds, qat_preds))
    / len(base_preds) * 100
)

print(f"\nFP32 대비 예측 일치율: {agreement:.2f}%")

# 7. 팀원 전달용 일반 FP32 가중치 모델 저장
# 주의: 실제 INT8 모델이 아니라 QAT 학습 가중치임
print("\n팀원 전달용 모델 생성 중...")

float_model = AutoModelForSequenceClassification.from_pretrained(
    str(BASE_DIR),
    local_files_only=True,
)

prepared_state = qat_model.state_dict()
float_state = float_model.state_dict()

missing = [
    key for key in float_state
    if key not in prepared_state
]

if missing:
    raise RuntimeError(
        f"가중치 변환 실패. 누락된 키: {missing[:10]}"
    )

for key in float_state:
    if float_state[key].shape != prepared_state[key].shape:
        raise RuntimeError(f"가중치 크기 불일치: {key}")

# Fake Quantization 모듈을 제외하고
# 원래 모델의 가중치 이름에 해당하는 텐서만 복사
with torch.no_grad():
    for key in float_state:
        float_state[key].copy_(prepared_state[key])

float_model.load_state_dict(float_state, strict=True)

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
float_model.save_pretrained(
    str(EXPORT_DIR),
    safe_serialization=True,
)
tokenizer.save_pretrained(str(EXPORT_DIR))

_, float_preds = predict(float_model)
float_metrics = calculate_metrics(y_true, float_preds)

print_metrics("QAT 학습 가중치 FP32", float_metrics)

# 8. 평가 결과 저장
results = {
    "dataset": "YuminKim/KoCoSa",
    "split": "test",
    "sample_count": len(y_true),
    "max_length": MAX_LENGTH,
    "baseline_pytorch_fp32": base_metrics,
    "qat_fake_quant": qat_metrics,
    "qat_weights_fp32": float_metrics,
    "agreement_vs_pytorch_fp32_percent": agreement,
    "fake_quantized_linear_layers": layer_count,
    "weight_quantization": "INT8 per-channel symmetric",
    "activation_quantization": "INT8 per-token asymmetric",
    "note": (
        "These are PyTorch test metrics, not final "
        "ONNX INT8 runtime metrics."
    ),
}

QAT_DIR.mkdir(parents=True, exist_ok=True)

with open(
    QAT_DIR / "qat_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, ensure_ascii=False, indent=2)

with open(
    QAT_DIR / "qat_test_predictions.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "index", "label", "fp32_pred",
        "qat_fake_pred", "qat_float_pred"
    ])

    for i in range(len(y_true)):
        writer.writerow([
            i,
            y_true[i],
            base_preds[i],
            qat_preds[i],
            float_preds[i],
        ])

print("\n===== 평가 완료 =====")
print("결과 저장 위치:", QAT_DIR)
print("팀원 전달용 가중치:", EXPORT_DIR)
