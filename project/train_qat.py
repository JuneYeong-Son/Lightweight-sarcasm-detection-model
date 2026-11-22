import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from sklearn.metrics import accuracy_score, f1_score

from torchao.quantization import quantize_
from torchao.quantization.qat import (
    QATConfig,
    IntxFakeQuantizeConfig,
    FakeQuantizedLinear
)


# ============================
# 1. 학습 설정
# ============================

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()

SMOKE_TEST = args.smoke_test

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = PROJECT_DIR.parent
SAVE_DIR = PROJECT_DIR / "qat_model"

EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
MAX_LENGTH = 256

torch.manual_seed(42)
device = torch.device("cpu")

print("학습 장치:", device, flush=True)
print("기존 모델 경로:", MODEL_NAME, flush=True)
print("결과 저장 경로:", SAVE_DIR, flush=True)
print("시험 학습 모드:", SMOKE_TEST, flush=True)


# ============================
# 2. 기존 모델 불러오기
# ============================

print("\n기존 파인튜닝 모델 로딩 중...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_NAME),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_NAME),
    local_files_only=True
)

model.to(device)

print("기존 모델 로딩 완료!", flush=True)

label_map = {
    "Non-Sarcasm": 0,
    "Sarcasm": 1
}


# ============================
# 3. 데이터셋 불러오기
# ============================

print("\nKoCoSa 데이터셋 로딩 중...", flush=True)

dataset = load_dataset("YuminKim/KoCoSa")

train_data = dataset["train"]
val_data = dataset["validation"]

# 시험 학습에서는 데이터 32개씩만 사용
if SMOKE_TEST:
    train_data = train_data.shuffle(seed=42).select(range(32))
    val_data = val_data.shuffle(seed=42).select(range(32))

print("학습 데이터 수:", len(train_data))
print("검증 데이터 수:", len(val_data))


# ============================
# 4. 데이터 전처리
# ============================

def preprocess(examples):
    result = tokenizer(
        examples["context"],
        examples["response"],
        truncation=True,
        max_length=MAX_LENGTH
    )

    result.pop("token_type_ids", None)

    result["labels"] = [
        label_map[label] for label in examples["label"]
    ]

    return result


train_data = train_data.map(
    preprocess,
    batched=True,
    remove_columns=train_data.column_names
)

val_data = val_data.map(
    preprocess,
    batched=True,
    remove_columns=val_data.column_names
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator
)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)


# ============================
# 5. 성능 평가 함수
# ============================

def evaluate(model):

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)

    f1 = f1_score(
        true_labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return accuracy, f1


# 기존 FP32 모델 성능 확인
print("\n기존 FP32 모델 평가 중...", flush=True)

base_accuracy, base_f1 = evaluate(model)

print(f"FP32 Accuracy: {base_accuracy:.4f}")
print(f"FP32 F1-score: {base_f1:.4f}")


# ============================
# 6. INT8 Fake Quantization 설정
# ============================

print("\nQAT 준비 중...", flush=True)

# Activation: INT8 Per-token, Asymmetric
activation_config = IntxFakeQuantizeConfig(
    torch.int8,
    "per_token",
    is_symmetric=False
)

# Weight: INT8 Per-channel, Symmetric
weight_config = IntxFakeQuantizeConfig(
    torch.int8,
    "per_channel",
    is_symmetric=True
)

qat_config = QATConfig(
    activation_config=activation_config,
    weight_config=weight_config,
    step="prepare"
)

# Linear 계층에 Fake Quantization 적용
quantize_(
    model,
    qat_config,
    filter_fn=lambda module, name: isinstance(
        module, torch.nn.Linear
    )
)

# Fake Quantization 적용 계층 수 확인
qat_layer_count = sum(
    isinstance(module, FakeQuantizedLinear)
    for module in model.modules()
)

print("Fake Quantization 적용 계층 수:", qat_layer_count)

if qat_layer_count == 0:
    raise RuntimeError("Fake Quantization이 적용되지 않았습니다.")

print("QAT 준비 완료!", flush=True)


# ============================
# 7. QAT 학습 전 성능 평가
# ============================

print("\nQAT 학습 전 모델 평가 중...", flush=True)

before_accuracy, before_f1 = evaluate(model)

print(f"Before QAT Accuracy: {before_accuracy:.4f}")
print(f"Before QAT F1-score: {before_f1:.4f}")


# ============================
# 8. Optimizer 설정
# ============================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================
# 9. QAT 추가 학습
# ============================

best_f1 = -1.0

for epoch in range(EPOCHS):

    print(f"\n===== QAT Epoch {epoch+1}/{EPOCHS} =====", flush=True)

    start_time = time.perf_counter()

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss).item():
            raise RuntimeError("Loss가 NaN 또는 Inf입니다.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True
        )

        optimizer.step()

        total_loss += loss.item()

        if SMOKE_TEST or step == 1 or step % 20 == 0:

            print(
                f"Step {step}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = total_loss / len(train_loader)


    # ============================
    # 10. Validation 평가
    # ============================

    accuracy, f1 = evaluate(model)

    elapsed = time.perf_counter() - start_time

    print(f"\nQAT Epoch {epoch+1} 결과")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")
    print(f"소요 시간: {elapsed/60:.1f}분", flush=True)


    # ============================
    # 11. 최고 QAT 모델 저장
    # ============================

    if f1 > best_f1:

        best_f1 = f1

        if not SMOKE_TEST:

            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            torch.save(
                model.state_dict(),
                SAVE_DIR / "best_qat_prepared.pt"
            )

            model.config.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

            print("최고 QAT 모델 저장 완료!", flush=True)


# ============================
# 12. 최종 결과 출력
# ============================

print("\n===== QAT 결과 비교 =====")

print(f"기존 FP32 F1-score: {base_f1:.4f}")
print(f"QAT 학습 전 F1-score: {before_f1:.4f}")
print(f"QAT 학습 후 F1-score: {best_f1:.4f}")

if SMOKE_TEST:

    print("\n===== QAT 시험 학습 완료 =====")
    print("시험 학습 모델은 저장하지 않습니다.")

else:

    print("\n===== QAT 전체 학습 완료 =====")
    print(f"모델 저장 위치: {SAVE_DIR}")
