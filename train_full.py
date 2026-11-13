
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

# ============================
# 1. 학습 설정
# ============================

MODEL_NAME = "klue/roberta-base"
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
MAX_LENGTH = 256

SAVE_DIR = Path.home() / "kocosa_full_model"

torch.manual_seed(42)
device = torch.device("cpu")

print("학습 장치:", device, flush=True)

# ============================
# 2. 데이터셋 불러오기
# ============================

print("KoCoSa 데이터셋 로딩 중...", flush=True)

dataset = load_dataset("YuminKim/KoCoSa")

label_map = {
    "Non-Sarcasm": 0,
    "Sarcasm": 1
}

# ============================
# 3. Tokenizer 및 전처리
# ============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    result = tokenizer(
        examples["context"],
        examples["response"],
        truncation=True,
        max_length=MAX_LENGTH
    )

    # KLUE-RoBERTa 호환성 처리
    result.pop("token_type_ids", None)

    result["labels"] = [
        label_map[label] for label in examples["label"]
    ]

    return result

train_data = dataset["train"].map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

val_data = dataset["validation"].map(
    preprocess,
    batched=True,
    remove_columns=dataset["validation"].column_names
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
# 4. 모델 준비
# ============================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Non-Sarcasm", 1: "Sarcasm"},
    label2id=label_map
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================
# 5. Full Fine-Tuning
# ============================

best_f1 = -1.0

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====", flush=True)

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

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if step == 1 or step % 20 == 0:
            print(
                f"Step {step}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = total_loss / len(train_loader)

    # ========================
    # 6. Validation 평가
    # ========================

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
        true_labels, predictions,
        average="binary",
        zero_division=0
    )

    elapsed = time.perf_counter() - start_time

    print(f"\nEpoch {epoch+1} 결과")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")
    print(f"소요 시간: {elapsed/60:.1f}분", flush=True)

    # 가장 좋은 모델 저장
    if f1 > best_f1:
        best_f1 = f1

        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        print("최고 성능 모델 저장 완료!", flush=True)

print("\n===== Fine-Tuning 완료 =====")
print(f"최고 Validation F1-score: {best_f1:.4f}")
print(f"모델 저장 위치: {SAVE_DIR}")
