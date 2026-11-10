"""KoCoSa 평가 지표 (논문 arXiv:2402.14428 Table 8과 같은 지표).

numpy만 사용해서 PyTorch 없이(ONNX만 설치된 환경에서도) 쓸 수 있다.
"""

import numpy as np

LABEL_MAP = {"Non-Sarcasm": 0, "Sarcasm": 1}

METRIC_KEYS = (
    "accuracy", "f1_sarcasm", "balanced_accuracy", "macro_f1", "weighted_f1",
    "precision_s", "recall_s", "precision_n", "recall_n",
)


def compute_metrics(labels, preds) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_class = {}
    for cls in (0, 1):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int((labels == cls).sum()),
        }
    total = len(labels)
    return {
        "accuracy": float((preds == labels).mean()),
        "f1_sarcasm": per_class[1]["f1"],  # train.py에서 쓰던 F1 (Sarcasm 기준 이진 F1)
        "balanced_accuracy": (per_class[0]["recall"] + per_class[1]["recall"]) / 2,
        "macro_f1": (per_class[0]["f1"] + per_class[1]["f1"]) / 2,
        "weighted_f1": sum(per_class[c]["f1"] * per_class[c]["support"] for c in (0, 1)) / total,
        "precision_s": per_class[1]["precision"],
        "recall_s": per_class[1]["recall"],
        "precision_n": per_class[0]["precision"],
        "recall_n": per_class[0]["recall"],
        "predicted_sarcasm_ratio": float((preds == 1).mean()),
    }
