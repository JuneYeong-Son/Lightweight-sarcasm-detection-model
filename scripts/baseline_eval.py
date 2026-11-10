"""비꼼 탐지 모델의 학습 전/후 성능을 같은 지표로 비교한다.

비교 대상
  1. 다수 클래스 기준선: 모든 입력을 Sarcasm으로 예측
  2. 학습 전: klue/roberta-base 원본 (분류층은 무작위 초기화, seed 여러 개)
  3. 학습 후: fine-tuned 모델 (model/)

지표는 KoCoSa 논문(arXiv:2402.14428) Table 8과 같은 것을 함께 계산한다.
    python scripts/baseline_eval.py
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# scripts/ 폴더의 파일이 같은 이름의 표준 모듈을 가리지 않도록, 스크립트 폴더를
# import 경로 맨 뒤로 옮긴다 (표준 모듈이 먼저, metrics.py는 그다음에 찾는다).
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]
sys.path.append(str(_SCRIPT_DIR))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from metrics import LABEL_MAP, METRIC_KEYS, compute_metrics  # noqa: E402

PROJECT_ROOT = _SCRIPT_DIR.parent
ID2LABEL = {0: "Non-Sarcasm", 1: "Sarcasm"}


def encode_batches(tokenizer, split, max_length, batch_size):
    data = load_dataset("YuminKim/KoCoSa", split=split)
    labels = [LABEL_MAP[label] for label in data["label"]]
    batches = []
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        # train.py 학습 때와 같은 방식: context/response 문장쌍, token_type_ids 제외
        enc = tokenizer(
            chunk["context"], chunk["response"],
            truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        batches.append((enc["input_ids"], enc["attention_mask"]))
    return labels, batches


@torch.inference_mode()
def predict(model, batches):
    model.eval()
    preds = []
    for input_ids, attention_mask in batches:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds.extend(logits.argmax(dim=1).tolist())
    return preds


def load_untrained(base_model, seed):
    # 분류층(classifier)은 원본에 없어서 무작위로 만들어진다. seed로 재현 가능하게 한다.
    torch.manual_seed(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL_MAP,
    )


def summarize(runs):
    out = {}
    for key in METRIC_KEYS + ("predicted_sarcasm_ratio",):
        values = [r[key] for r in runs]
        out[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return out


PAPER_COLUMNS = (  # 논문 Table 8 순서
    ("BA", "balanced_accuracy"), ("M-F1", "macro_f1"), ("W-F1", "weighted_f1"),
    ("Precision-S", "precision_s"), ("Recall-S", "recall_s"),
    ("Precision-N", "precision_n"), ("Recall-N", "recall_n"),
    ("Acc", "accuracy"), ("F1(S)", "f1_sarcasm"),
)


def cell(value):
    # 논문처럼 백분율 소수 첫째 자리, 여러 번 돌린 값은 평균 (±표준편차)
    if isinstance(value, dict):
        return f"{value['mean'] * 100:.1f} (±{value['std'] * 100:.1f})"
    return f"{value * 100:.1f}"


def print_table(split, rows):
    # Markdown 표로 출력해서 문서에 그대로 옮길 수 있게 한다.
    print(f"\n===== {split} =====")
    print("| Model | " + " | ".join(h for h, _ in PAPER_COLUMNS) + " |")
    print("|---" * (len(PAPER_COLUMNS) + 1) + "|")
    for name, metrics in rows:
        print(f"| {name} | " + " | ".join(cell(metrics[k]) for _, k in PAPER_COLUMNS) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"),
                        default=["validation", "test"])
    parser.add_argument("--base-model", default="klue/roberta-base")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="학습 전 모델의 분류층 초기화 seed")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "baseline_eval.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    finetuned = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "model_dir": str(args.model_dir.relative_to(PROJECT_ROOT)),
        "seeds": args.seeds,
        "splits": {},
    }

    for split in args.splits:
        labels, batches = encode_batches(tokenizer, split, args.max_length, args.batch_size)
        print(f"\n[{split}] {len(labels)}개 (Sarcasm {sum(labels)}, "
              f"Non-Sarcasm {len(labels) - sum(labels)})", flush=True)

        majority = compute_metrics(labels, [1] * len(labels))

        untrained_runs = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            model = load_untrained(args.base_model, seed)
            metrics = compute_metrics(labels, predict(model, batches))
            metrics["seed"] = seed
            untrained_runs.append(metrics)
            del model
            print(f"[{split}] 학습 전 seed {seed}: 정확도 {metrics['accuracy']:.4f}, "
                  f"Sarcasm 예측 비율 {metrics['predicted_sarcasm_ratio']:.2f} "
                  f"({time.perf_counter() - t0:.0f}초)", flush=True)

        t0 = time.perf_counter()
        after = compute_metrics(labels, predict(finetuned, batches))
        print(f"[{split}] 학습 후: 정확도 {after['accuracy']:.4f} "
              f"({time.perf_counter() - t0:.0f}초)", flush=True)

        untrained_summary = summarize(untrained_runs)
        report["splits"][split] = {
            "samples": len(labels),
            "sarcasm": sum(labels),
            "majority_baseline": majority,
            "untrained": {"runs": untrained_runs, "summary": untrained_summary},
            "finetuned": after,
        }
        print_table(split, [
            ("다수 클래스 (전부 Sarcasm)", majority),
            (f"학습 전 (seed {len(args.seeds)}개 평균)", untrained_summary),
            ("학습 후 (fine-tuned)", after),
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
