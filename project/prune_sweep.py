import time
import json
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "klue/roberta-base"
MAX_LENGTH = 256

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
FT_MODEL_DIR = MODELS_DIR 
PRUNE_SAVE_DIR = MODELS_DIR / "pruned"
PRUNE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 테스트할 프루닝 비율 (선행논문과 동일: 10%~90%, 10%씩 9개)
PRUNE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

device = torch.device("cpu")
print("디바이스:", device, flush=True)

print("KoCoSa validation 데이터셋 로딩 중...", flush=True)
dataset = load_dataset("YuminKim/KoCoSa")

label_map = {
    "Non-Sarcasm": 0,
    "Sarcasm": 1
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    result = tokenizer(
        examples["context"],
        examples["response"],
        truncation=True,
        max_length=MAX_LENGTH
    )

    # KLUE-RoBERTa 호환성 처리 (train_full.py와 동일)
    result.pop("token_type_ids", None)

    result["labels"] = [
        label_map[label] for label in examples["label"]
    ]

    return result

val_data = dataset["validation"].map(
    preprocess,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)


def evaluate(model):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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
    return accuracy, f1

# 선행논문(Gordon et al., 2020) 기준: 인코더 레이어별 6개 행렬

def get_prunable_modules(model):
    modules = []

    # word embedding
    modules.append((model.roberta.embeddings.word_embeddings, "weight"))

    # 인코더 레이어별 6개 Linear 행렬
    for layer in model.roberta.encoder.layer:
        modules.append((layer.attention.self.query, "weight"))
        modules.append((layer.attention.self.key, "weight"))
        modules.append((layer.attention.self.value, "weight"))
        modules.append((layer.attention.output.dense, "weight"))
        modules.append((layer.intermediate.dense, "weight"))
        modules.append((layer.output.dense, "weight"))

    return modules

def apply_pruning(model, ratio):
    """비구조적(unstructured) magnitude(L1) 프루닝을 행렬별로(matrix-local) 적용.
    각 행렬 안에서 절댓값 기준 하위 ratio만큼을 0으로 만들고,
    마스크를 weight에 영구 반영(remove)한다."""
    modules = get_prunable_modules(model)
    for module, param_name in modules:
        prune.l1_unstructured(module, name=param_name, amount=ratio)
        prune.remove(module, param_name)  # 마스크를 실제 weight에 영구 반영
    return model

def density(model):
    """0이 아닌 파라미터 비율 (용량 비교용 참고 지표 - 비구조적 프루닝은
    파일 크기 자체는 줄지 않으므로, 실질적으로 얼마나 잘렸는지는 이 값으로 확인)"""
    modules = get_prunable_modules(model)
    total, nonzero = 0, 0
    for module, param_name in modules:
        w = getattr(module, param_name)
        total += w.numel()
        nonzero += (w != 0).sum().item()
    return nonzero / total

results = []

for ratio in PRUNE_RATIOS:
    print(f"\n===== 프루닝 비율 {int(ratio*100)}% =====", flush=True)
    start = time.perf_counter()

    model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL_DIR)
    model.to(device)

    model = apply_pruning(model, ratio)

    acc, f1 = evaluate(model)
    dens = density(model)

    elapsed = time.perf_counter() - start

    print(
        f"Accuracy: {acc:.4f} | F1: {f1:.4f} | "
        f"밀도(0이 아닌 비율): {dens:.4f} | 소요: {elapsed:.1f}초",
        flush=True
    )

    results.append({
        "prune_ratio": ratio,
        "accuracy": acc,
        "f1": f1,
        "density": dens
    })

    # 비율별 모델 저장 (나중에 추가 학습/양자화에 그대로 이어서 쓸 수 있도록)
    save_path = PRUNE_SAVE_DIR / f"pruned_{int(ratio * 100)}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# 전체 결과 요약 출력 + 저장

print("\n===== 전체 결과 =====")
print(f"{'비율':>6} | {'Accuracy':>9} | {'F1':>7} | {'밀도':>7}")
for r in results:
    print(
        f"{int(r['prune_ratio'] * 100):>5}% | "
        f"{r['accuracy']:>9.4f} | {r['f1']:>7.4f} | {r['density']:>7.4f}"
    )

with open(PRUNE_SAVE_DIR / "prune_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n결과표 저장 위치: {PRUNE_SAVE_DIR / 'prune_results.json'}")
print(f"비율별 모델 저장 위치: {PRUNE_SAVE_DIR}/pruned_<비율>")
