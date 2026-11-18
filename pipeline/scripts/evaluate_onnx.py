"""프루닝 FP32 ONNX와 동적 INT8 ONNX의 정확도/F1/크기/지연시간을 KoCoSa로 비교한다.

venv가 아니라 기본 Python(datasets 사용 가능)으로 실행한다.
    python pipeline/scripts/evaluate_onnx.py --split validation
    python pipeline/scripts/evaluate_onnx.py --split test
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import AutoTokenizer

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PIPELINE_DIR.parent
LABEL_MAP = {"Non-Sarcasm": 0, "Sarcasm": 1}


def session(path: Path, threads: int):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def encode(tokenizer, ctx, resp, max_length):
    enc = tokenizer(ctx, resp, truncation=True, max_length=max_length,
                    padding=True, return_tensors="np")
    return {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}


def evaluate(name, sess, tokenizer, data, args):
    ctx, resp, labels = data
    preds = []
    for i in range(0, len(labels), args.batch_size):
        feed = encode(tokenizer, ctx[i:i + args.batch_size], resp[i:i + args.batch_size],
                      args.max_length)
        preds.extend(sess.run(["logits"], feed)[0].argmax(1).tolist())

    samples = [encode(tokenizer, [c], [r], args.max_length)
               for c, r in zip(ctx[:args.latency_samples], resp[:args.latency_samples])]
    for i in range(5):
        sess.run(["logits"], samples[i % len(samples)])
    lat = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        sess.run(["logits"], samples[i % len(samples)])
        lat.append((time.perf_counter() - t0) * 1000)

    result = {
        "accuracy": accuracy_score(labels, preds),
        "f1_sarcasm": f1_score(labels, preds, pos_label=1),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "latency_ms_mean_batch1": statistics.mean(lat),
        "latency_ms_median_batch1": statistics.median(lat),
    }
    print(f"[{name}] " + ", ".join(f"{k} {v:.4f}" for k, v in result.items()), flush=True)
    return result, preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp32-dir", type=Path, default=PIPELINE_DIR / "3_onnx_fp32")
    parser.add_argument("--int8-dir", type=Path, default=PROJECT_ROOT / "onnx-pruned-int8")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-samples", type=int, default=32)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    ds = load_dataset("YuminKim/KoCoSa", split=args.split)
    data = (list(ds["context"]), list(ds["response"]), [LABEL_MAP[x] for x in ds["label"]])
    tokenizer = AutoTokenizer.from_pretrained(args.fp32_dir, local_files_only=True)

    report = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              "split": args.split, "samples": len(data[2]), "threads": args.threads,
              "onnxruntime": ort.__version__, "results": {}}
    all_preds = {}
    for name, directory in (("pruned-fp32", args.fp32_dir), ("pruned-int8", args.int8_dir)):
        path = directory.resolve() / "model.onnx"
        res, preds = evaluate(name, session(path, args.threads), tokenizer, data, args)
        res["file_size_mb"] = path.stat().st_size / 1024 ** 2
        report["results"][name] = res
        all_preds[name] = np.asarray(preds)
    report["prediction_agreement"] = float(
        (all_preds["pruned-fp32"] == all_preds["pruned-int8"]).mean())
    print(f"예측 일치율: {report['prediction_agreement']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"pruned_dynint8_{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
