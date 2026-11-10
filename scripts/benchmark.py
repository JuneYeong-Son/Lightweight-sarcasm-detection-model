"""PyTorch 원본 모델과 ONNX 모델의 추론 성능을 같은 조건에서 비교한다.

측정 항목: 모델 로드 시간, 추론 지연시간(batch 1), 메모리, 파일 크기, 정확도/F1.
백엔드마다 별도 프로세스에서 실행해서 메모리 측정이 서로 섞이지 않게 한다.

PC와 Android(Termux + proot-distro Ubuntu)에서 같은 방식으로 실행한다.
    python scripts/benchmark.py
    python scripts/benchmark.py --no-eval --threads 4
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# scripts/ 폴더의 파일(예: code.py)이 같은 이름의 표준 모듈을 가리지 않도록
# 스크립트 폴더를 import 경로에서 뺀다.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP = {"Non-Sarcasm": 0, "Sarcasm": 1}
BACKENDS = ("pytorch", "onnx")


# ============================
# 메모리 측정 (추가 패키지 없이)
# ============================

def _linux_status_mb(field: str):
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return None


def _windows_memory_mb():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
    ]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        return None, None
    mb = 1024 * 1024
    return counters.WorkingSetSize / mb, counters.PeakWorkingSetSize / mb


def current_rss_mb():
    if sys.platform.startswith("linux"):
        return _linux_status_mb("VmRSS")
    if sys.platform == "win32":
        return _windows_memory_mb()[0]
    return None


def peak_rss_mb():
    if sys.platform.startswith("linux"):
        return _linux_status_mb("VmHWM")
    if sys.platform == "win32":
        return _windows_memory_mb()[1]
    try:
        import resource  # macOS: bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return None


# ============================
# 데이터
# ============================

def load_split(split: str, limit):
    from datasets import load_dataset

    data = load_dataset("YuminKim/KoCoSa", split=split)
    if limit is not None:
        data = data.select(range(min(limit, len(data))))
    return (
        list(data["context"]),
        list(data["response"]),
        [LABEL_MAP[label] for label in data["label"]],
    )


def encode(tokenizer, contexts, responses, max_length):
    # 코드.py 학습 때와 같은 방식: context/response 문장쌍, token_type_ids 제외
    enc = tokenizer(
        contexts,
        responses,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


# ============================
# 백엔드
# ============================

class TorchRunner:
    def __init__(self, model_dir: Path, threads: int):
        import torch
        from transformers import AutoModelForSequenceClassification

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
        self.model.eval()

    def __call__(self, input_ids, attention_mask):
        with self.torch.inference_mode():
            return self.model(
                input_ids=self.torch.from_numpy(input_ids),
                attention_mask=self.torch.from_numpy(attention_mask),
            ).logits.numpy()


class OnnxRunner:
    def __init__(self, onnx_path: Path, threads: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, input_ids, attention_mask):
        return self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]


def model_files(backend: str, args):
    if backend == "pytorch":
        return sorted(args.model_dir.glob("*.safetensors"))
    onnx_path = args.onnx_dir / "model.onnx"
    return [onnx_path] + sorted(args.onnx_dir.glob("model.onnx.data"))


def library_versions(backend: str):
    import transformers

    versions = {"numpy": np.__version__, "transformers": transformers.__version__}
    if backend == "pytorch":
        import torch
        versions["torch"] = torch.__version__
    else:
        import onnxruntime
        versions["onnxruntime"] = onnxruntime.__version__
    return versions


# ============================
# 백엔드 1개 측정 (자식 프로세스)
# ============================

def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def run_backend(backend: str, args) -> dict:
    from transformers import AutoTokenizer

    log = lambda msg: print(f"[{backend}] {msg}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    sample_count = args.latency_samples
    eval_limit = args.eval_limit
    if args.no_eval:
        load_limit = sample_count
    elif eval_limit is None:
        load_limit = None
    else:
        load_limit = max(sample_count, eval_limit)
    contexts, responses, labels = load_split(args.split, load_limit)

    rss_before = current_rss_mb()
    start = time.perf_counter()
    if backend == "pytorch":
        runner = TorchRunner(args.model_dir, args.threads)
    else:
        runner = OnnxRunner(args.onnx_dir / "model.onnx", args.threads)
    load_seconds = time.perf_counter() - start
    rss_after_load = current_rss_mb()
    log(f"모델 로드 {load_seconds:.1f}초")

    # ---- 지연시간: batch 1, 실제 데이터 앞쪽 N개를 돌아가며 사용 ----
    samples = [
        encode(tokenizer, [c], [r], args.max_length)
        for c, r in zip(contexts[:sample_count], responses[:sample_count])
    ]
    seq_lens = [int(ids.shape[1]) for ids, _ in samples]

    for i in range(args.warmup):
        runner(*samples[i % len(samples)])

    latencies_ms = []
    for i in range(args.runs):
        ids, mask = samples[i % len(samples)]
        t0 = time.perf_counter()
        runner(ids, mask)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    log(f"지연시간 평균 {statistics.mean(latencies_ms):.1f} ms ({args.runs}회)")

    result = {
        "backend": backend,
        "files": [str(p.relative_to(PROJECT_ROOT)) for p in model_files(backend, args)],
        "file_size_mb": sum(p.stat().st_size for p in model_files(backend, args)) / 1024**2,
        "load_seconds": load_seconds,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p90": percentile(latencies_ms, 90),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "std": statistics.pstdev(latencies_ms),
        },
        "latency_inputs": {
            "samples": len(samples),
            "seq_len_mean": statistics.mean(seq_lens),
            "seq_len_max": max(seq_lens),
        },
        "versions": library_versions(backend),
    }

    # ---- 정확도/F1: 데이터셋 전체(또는 --eval-limit개) ----
    if not args.no_eval:
        if eval_limit is not None:
            contexts, responses, labels = (
                contexts[:eval_limit], responses[:eval_limit], labels[:eval_limit]
            )
        predictions, all_logits = [], []
        total = len(labels)
        t0 = time.perf_counter()
        for start_idx in range(0, total, args.batch_size):
            end_idx = min(start_idx + args.batch_size, total)
            ids, mask = encode(
                tokenizer,
                contexts[start_idx:end_idx],
                responses[start_idx:end_idx],
                args.max_length,
            )
            logits = runner(ids, mask)
            all_logits.append(logits)
            predictions.extend(logits.argmax(axis=1).tolist())
            done_batches = start_idx // args.batch_size + 1
            if done_batches % 20 == 0 or end_idx == total:
                log(f"평가 {end_idx}/{total}")
        eval_seconds = time.perf_counter() - t0

        labels_arr = np.asarray(labels)
        preds_arr = np.asarray(predictions)
        tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
        fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
        fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        result["eval"] = {
            "split": args.split,
            "samples": total,
            "accuracy": float((preds_arr == labels_arr).mean()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "seconds": eval_seconds,
            "samples_per_second": total / eval_seconds,
        }
        result["_predictions"] = predictions
        result["_logits"] = np.concatenate(all_logits).tolist()
        log(f"정확도 {result['eval']['accuracy']:.4f}, F1 {f1:.4f}")

    result["memory_mb"] = {
        "before_load": rss_before,
        "after_load": rss_after_load,
        "model_load_increase": (
            rss_after_load - rss_before
            if rss_before is not None and rss_after_load is not None
            else None
        ),
        "peak": peak_rss_mb(),
    }
    return result


# ============================
# 전체 실행 (부모 프로세스)
# ============================

def device_info(name):
    info = {
        "name": name or platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    # Android(Termux/proot)에서는 getprop으로 기종을 알 수 있으면 기록
    for key, prop in (("android_model", "ro.product.model"), ("android_soc", "ro.soc.model")):
        try:
            value = subprocess.run(
                ["getprop", prop], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if value:
                info[key] = value
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def pad(text, width):
    # 한글처럼 화면에서 두 칸을 차지하는 글자를 고려해 왼쪽 정렬
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def print_summary(results):
    rows = [
        ("파일 크기 (MB)", lambda r: fmt(r["file_size_mb"], ".1f")),
        ("모델 로드 (초)", lambda r: fmt(r["load_seconds"], ".2f")),
        ("지연시간 평균 (ms)", lambda r: fmt(r["latency_ms"]["mean"], ".1f")),
        ("지연시간 중앙값 (ms)", lambda r: fmt(r["latency_ms"]["median"], ".1f")),
        ("지연시간 p90 (ms)", lambda r: fmt(r["latency_ms"]["p90"], ".1f")),
        ("로드 후 메모리 증가 (MB)", lambda r: fmt(r["memory_mb"]["model_load_increase"], ".0f")),
        ("최대 메모리 (MB)", lambda r: fmt(r["memory_mb"]["peak"], ".0f")),
        ("정확도", lambda r: fmt(r.get("eval", {}).get("accuracy"), ".4f")),
        ("F1", lambda r: fmt(r.get("eval", {}).get("f1"), ".4f")),
        ("평가 처리량 (개/초)", lambda r: fmt(r.get("eval", {}).get("samples_per_second"), ".2f")),
    ]
    names = [r["backend"] for r in results]
    print("\n===== 결과 =====")
    print(pad("항목", 26) + "".join(f"{n:>14}" for n in names))
    for label, getter in rows:
        print(pad(label, 26) + "".join(f"{getter(r):>14}" for r in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--onnx-dir", type=Path, default=PROJECT_ROOT / "onnx")
    parser.add_argument("--threads", type=int, default=4,
                        help="추론 스레드 수 (기본 4). 폰은 큰 코어 수에 맞추는 것이 보통 빠르다.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--latency-samples", type=int, default=32,
                        help="지연시간 측정에 쓸 데이터 개수 (앞에서부터)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, help="정확도 평가 batch 크기")
    parser.add_argument("--no-eval", action="store_true", help="정확도/F1 평가 생략")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="정확도 평가에 쓸 최대 데이터 개수 (기본: 전체)")
    parser.add_argument("--cooldown", type=float, default=10,
                        help="백엔드 사이 대기 시간(초). 폰 발열 영향을 줄인다.")
    parser.add_argument("--device-name", default=None, help="결과 파일에 남길 기기 이름")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "benchmark")
    parser.add_argument("--child", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.onnx_dir = args.onnx_dir.resolve()

    if args.child:
        result = run_backend(args.child, args)
        args.child_output.write_text(json.dumps(result), encoding="utf-8")
        return

    for backend in args.backends:
        missing = [p for p in model_files(backend, args) if not p.exists()]
        if missing or not model_files(backend, args):
            parser.error(f"{backend} 모델 파일이 없습니다: {missing or args.model_dir}")

    child_args = [a for a in sys.argv[1:]]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, backend in enumerate(args.backends):
            if i > 0 and args.cooldown > 0:
                print(f"\n{args.cooldown:.0f}초 대기 (발열 완화)...", flush=True)
                time.sleep(args.cooldown)
            print(f"\n===== {backend} 측정 시작 =====", flush=True)
            out = Path(tmp) / f"{backend}.json"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *child_args,
                 "--child", backend, "--child-output", str(out)],
                check=True,
            )
            results.append(json.loads(out.read_text(encoding="utf-8")))

    # 백엔드 간 예측 일치율 (같은 데이터로 평가했을 때)
    agreement = None
    evaluated = [r for r in results if "_predictions" in r]
    if len(evaluated) == 2:
        a, b = evaluated
        same = np.mean(np.asarray(a["_predictions"]) == np.asarray(b["_predictions"]))
        diff = np.abs(np.asarray(a["_logits"]) - np.asarray(b["_logits"])).max()
        agreement = {
            "backends": [a["backend"], b["backend"]],
            "prediction_agreement": float(same),
            "max_logit_diff": float(diff),
        }
    for r in results:
        r.pop("_predictions", None)
        r.pop("_logits", None)

    print_summary(results)
    if agreement:
        print(f"\n예측 일치율: {agreement['prediction_agreement']:.4f}, "
              f"최대 logit 차이: {agreement['max_logit_diff']:.2e}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device_info(args.device_name),
        "settings": {
            "threads": args.threads,
            "split": args.split,
            "max_length": args.max_length,
            "latency_samples": args.latency_samples,
            "warmup": args.warmup,
            "runs": args.runs,
            "batch_size": args.batch_size,
            "eval": not args.no_eval,
            "eval_limit": args.eval_limit,
        },
        "results": results,
        "agreement": agreement,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in report["device"]["name"])
    out_path = args.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{safe_name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
