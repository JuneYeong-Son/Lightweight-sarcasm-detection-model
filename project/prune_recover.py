import time
import json
import copy
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

# ============================
# 1. 설정
# ============================

MODEL_NAME = "klue/roberta-base"
MAX_LENGTH = 256

# train_full.py와 동일한 하이퍼파라미터 (회복 학습도 같은 설정으로 진행)
BATCH_SIZE = 16
LEARNING_RATE = 1e-5

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# 파인튜닝된 모델이 models 폴더에 바로 들어있는 구조 (kocosa_full_model 서브폴더 없음)
FT_MODEL_DIR = MODELS_DIR

# 프루닝+회복학습 결과를 저장할 위치
RECOVER_SAVE_DIR = MODELS_DIR / "pruned_recovered"
RECOVER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 테스트할 프루닝 비율 (논문과 동일: 10%~90%, 10%씩 9개)
PRUNE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def get_recovery_epochs(ratio):
    """논문 기준: 기본 5epoch, 80%/90%는 7epoch"""
    if ratio >= 0.8:
        return 7
    return 5

device = torch.device("cpu")
print("디바이스:", device, flush=True)

# ============================
# 2. 데이터셋 (이번엔 train도 필요 - 회복 학습용)
# ============================

print("KoCoSa 데이터셋 로딩 중...", flush=True)
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
# 3. 평가 함수
# ============================

def evaluate(model):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="binary", zero_division=0)
    return accuracy, f1

# ============================
# 4. 프루닝 대상 모듈 (prune_sweep.py와 동일)
# ============================

def get_prunable_modules(model):
    modules = []
    modules.append((model.roberta.embeddings.word_embeddings, "weight"))
    for layer in model.roberta.encoder.layer:
        modules.append((layer.attention.self.query, "weight"))
        modules.append((layer.attention.self.key, "weight"))
        modules.append((layer.attention.self.value, "weight"))
        modules.append((layer.attention.output.dense, "weight"))
        modules.append((layer.intermediate.dense, "weight"))
        modules.append((layer.output.dense, "weight"))
    return modules

def apply_pruning_mask(model, ratio):
    """프루닝 마스크를 적용하되, 아직 영구 반영(remove)하지 않음.
    이 상태로 두면 매 forward마다 마스크가 자동으로 다시 곱해져서,
    재학습 중에도 0이었던 가중치는 계산에서 계속 0으로 취급된다."""
    for module, param_name in get_prunable_modules(model):
        prune.l1_unstructured(module, name=param_name, amount=ratio)
    return model

def make_pruning_permanent(model):
    """회복 학습이 끝난 뒤, 마스크를 실제 weight에 영구 반영."""
    for module, param_name in get_prunable_modules(model):
        prune.remove(module, param_name)
    return model

def density(model):
    total, nonzero = 0, 0
    for module, param_name in get_prunable_modules(model):
        w = getattr(module, param_name)
        total += w.numel()
        nonzero += (w != 0).sum().item()
    return nonzero / total

# ============================
# 5. 비율별: 프루닝 -> 회복 학습(N epoch) -> 제일 좋았던 epoch 선택 -> 영구 반영 -> 저장
# ============================

results = []

for ratio in PRUNE_RATIOS:
    epochs = get_recovery_epochs(ratio)
    print(f"\n===== 프루닝 {int(ratio*100)}% + 회복학습 {epochs}epoch =====", flush=True)
    ratio_start = time.perf_counter()

    # 매 비율마다 파인튜닝된 원본 모델을 새로 불러옴 (비율 간 독립성 보장)
    model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL_DIR)
    model.to(device)

    # 프루닝 마스크 적용 (아직 remove() 호출 안 함 - 학습 중 마스크 유지 목적)
    model = apply_pruning_mask(model, ratio)

    # 프루닝 직후(학습 전) 성능도 참고용으로 기록
    acc0, f1_0 = evaluate(model)
    print(f"[회복학습 전] Accuracy: {acc0:.4f} | F1: {f1_0:.4f}", flush=True)

    # 옵티마이저는 프루닝 적용 "이후"에 만들어야 함
    # (프루닝하면 weight -> weight_orig로 파라미터 자체가 바뀌기 때문)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = f1_0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if step == 1 or step % 100 == 0:
                print(f"  epoch {epoch+1}/{epochs} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        acc, f1 = evaluate(model)
        elapsed = time.perf_counter() - epoch_start

        print(
            f"[epoch {epoch+1}/{epochs}] train_loss {avg_loss:.4f} | "
            f"Accuracy {acc:.4f} | F1 {f1:.4f} | 소요 {elapsed/60:.1f}분",
            flush=True
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            print("  -> 이번 epoch이 최고 성능, 상태 저장", flush=True)

    # 이 비율에서 제일 좋았던 epoch 상태로 복원
    model.load_state_dict(best_state)

    # 이제서야 마스크를 weight에 영구 반영 (더 이상 학습 안 하므로 안전)
    model = make_pruning_permanent(model)

    final_acc, final_f1 = evaluate(model)
    dens = density(model)
    ratio_elapsed = time.perf_counter() - ratio_start

    print(
        f"[{int(ratio*100)}% 최종] Accuracy {final_acc:.4f} | F1 {final_f1:.4f} | "
        f"밀도 {dens:.4f} | 총 소요 {ratio_elapsed/60:.1f}분",
        flush=True
    )

    results.append({
        "prune_ratio": ratio,
        "recovery_epochs": epochs,
        "accuracy_before_recovery": acc0,
        "f1_before_recovery": f1_0,
        "accuracy_after_recovery": final_acc,
        "f1_after_recovery": final_f1,
        "density": dens
    })

    save_path = RECOVER_SAVE_DIR / f"pruned_recovered_{int(ratio * 100)}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 비율 하나가 끝날 때마다 결과표를 바로 덮어써서 저장
    # (9개 다 끝나기 전에 중간에 멈춰도 그때까지의 결과는 남도록)
    with open(RECOVER_SAVE_DIR / "prune_recover_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> 결과표 갱신: {RECOVER_SAVE_DIR / 'prune_recover_results.json'}", flush=True)

# ============================
# 6. 전체 결과 요약 (콘솔 출력용, 저장은 이미 비율마다 끝났음)
# ============================

print("\n===== 전체 결과 (회복학습 전 -> 후) =====")
print(f"{'비율':>6} | {'Acc(전)':>8} | {'F1(전)':>8} | {'Acc(후)':>8} | {'F1(후)':>8} | {'밀도':>7}")
for r in results:
    print(
        f"{int(r['prune_ratio'] * 100):>5}% | "
        f"{r['accuracy_before_recovery']:>8.4f} | {r['f1_before_recovery']:>8.4f} | "
        f"{r['accuracy_after_recovery']:>8.4f} | {r['f1_after_recovery']:>8.4f} | "
        f"{r['density']:>7.4f}"
    )

print(f"\n결과표 저장 위치: {RECOVER_SAVE_DIR / 'prune_recover_results.json'}")
print(f"비율별 모델 저장 위치: {RECOVER_SAVE_DIR}/pruned_recovered_<비율>")
