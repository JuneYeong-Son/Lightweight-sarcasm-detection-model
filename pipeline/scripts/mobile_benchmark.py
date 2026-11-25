"""모바일(Android, Termux + proot-distro Ubuntu)에서 ONNX 모델 성능을 측정한다.

onnxruntime, tokenizers, numpy만 사용한다 (transformers, datasets, PyTorch 불필요).
PC에서도 같은 명령으로 실행해서 PC 결과를 기준(reference)으로 폰 결과와 비교한다.

측정 항목
  - 모델 로드 시간, 첫 추론 시간(cold start)
  - 지연시간 (batch 1): 토크나이저 / 추론 / 전체(end-to-end)
  - 메모리 (로드로 늘어난 RSS, 프로세스 최대 RSS)
  - 정확도 / F1 / BA / Macro-F1 (batch 1, 앱과 같은 조건)
  - PC 결과와의 예측 일치율, 최대 logit 차이 (--reference)
  - 지속 부하 (--sustained-seconds): 시간에 따른 지연시간, 온도, CPU 클럭 -> 발열 스로틀링

사용 순서
  1) PC: 평가 데이터를 파일로 저장 (datasets 필요, 한 번만)
       python pipeline/scripts/mobile_benchmark.py prepare
  2) PC: 기준 결과 만들기 (logits가 .npz로 함께 저장된다)
       python pipeline/scripts/mobile_benchmark.py run --device-name pc
  3) 폰: 같은 설정으로 측정하고 PC 결과와 비교
       python pipeline/scripts/mobile_benchmark.py run --device-name galaxy-s24 \\
           --reference results/mobile/<PC 결과>.npz --sustained-seconds 120
"""

import argparse
import glob
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LABEL_MAP = {"Non-Sarcasm": 0, "Sarcasm": 1}
DEFAULT_MODELS = (
    f"pruned-fp32={PROJECT_ROOT / 'pipeline' / '3_onnx_fp32'}",
    f"pruned-int8={PROJECT_ROOT / 'onnx-pruned-int8'}",
)


# ============================
# 기기 상태 (Linux/Android sysfs, 읽을 수 없으면 None)
# ============================

def _read(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _linux_status_mb(field):
    text = _read("/proc/self/status")
    if text:
        for line in text.splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    return None


def _windows_memory_mb():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                           wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None, None
    return c.WorkingSetSize / 1024 ** 2, c.PeakWorkingSetSize / 1024 ** 2


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
    return None


def cpu_freqs_mhz(kind="scaling_cur_freq"):
    """코어별 CPU 클럭(MHz). 폰은 big.LITTLE 구조라 코어마다 다르다."""
    freqs = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"),
                       key=lambda p: int(p.rsplit("cpu", 1)[1])):
        value = _read(f"{path}/cpufreq/{kind}")
        freqs.append(int(value) // 1000 if value and value.isdigit() else None)
    return freqs if any(f is not None for f in freqs) else None


def temperatures_c():
    """읽을 수 있는 온도 센서 중 최고값과 배터리 온도(°C). Android 버전에 따라 막혀 있을 수 있다."""
    zone_temps = []
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        value = _read(f"{zone}/temp")
        try:
            temp = int(value)
        except (TypeError, ValueError):
            continue
        temp = temp / 1000 if abs(temp) > 1000 else float(temp)  # 밀리°C 또는 °C
        if -20 < temp < 150:
            zone_temps.append(temp)
    battery = _read("/sys/class/power_supply/battery/temp")
    return {
        "max_zone": max(zone_temps) if zone_temps else None,
        "battery": int(battery) / 10 if battery and battery.lstrip("-").isdigit() else None,
    }


def battery_info():
    base = "/sys/class/power_supply/battery"
    capacity, status = _read(f"{base}/capacity"), _read(f"{base}/status")
    return {"capacity_percent": int(capacity) if capacity and capacity.isdigit() else None,
            "status": status}


def getprop(prop):
    try:
        return subprocess.run(["getprop", prop], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def device_info(name):
    import onnxruntime as ort
    import tokenizers

    info = {
        "name": name or platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_max_mhz": cpu_freqs_mhz("cpuinfo_max_freq"),
        "python": platform.python_version(),
        "versions": {"onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__,
                     "numpy": np.__version__},
        "battery": battery_info(),
        "temperature_c": temperatures_c(),
    }
    for key, prop in (("android_model", "ro.product.model"), ("android_soc", "ro.soc.model"),
                      ("android_version", "ro.build.version.release")):
        value = getprop(prop)
        if value:
            info[key] = value
    return info


# ============================
# 데이터, 지표
# ============================

def data_path(split):
    return DATA_DIR / f"kocosa_{split}.jsonl"


def prepare(args):
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        ds = load_dataset("YuminKim/KoCoSa", split=split)
        out = data_path(split)
        with out.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({"context": row["context"], "response": row["response"],
                                    "label": LABEL_MAP[row["label"]]}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(ds)}개 -> {out}")


def load_data(split, limit):
    path = data_path(split)
    if not path.exists():
        sys.exit(f"평가 데이터가 없습니다: {path}\nPC에서 먼저 'mobile_benchmark.py prepare'를 실행하고 "
                 f"data/ 폴더를 폰으로 옮기세요.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def compute_metrics(labels, preds):
    labels, preds = np.asarray(labels), np.asarray(preds)
    per = {}
    for cls in (0, 1):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[cls] = {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "n": int((labels == cls).sum())}
    return {
        "accuracy": float((preds == labels).mean()),
        "f1_sarcasm": per[1]["f1"],
        "balanced_accuracy": (per[0]["r"] + per[1]["r"]) / 2,
        "macro_f1": (per[0]["f1"] + per[1]["f1"]) / 2,
        "weighted_f1": sum(per[c]["f1"] * per[c]["n"] for c in (0, 1)) / len(labels),
        "precision_s": per[1]["p"], "recall_s": per[1]["r"],
        "precision_n": per[0]["p"], "recall_n": per[0]["r"],
        "predicted_sarcasm_ratio": float((preds == 1).mean()),
    }


def stats_ms(values):
    arr = np.asarray(values)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()), "max": float(arr.max()), "std": float(arr.std())}


# ============================
# 모델 1개 측정 (자식 프로세스)
# ============================

class Model:
    def __init__(self, model_dir: Path, threads: int, max_length: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # 학습과 같은 조건: 최대 256 토큰, 긴 쪽부터 자르기, token_type_ids 사용 안 함
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def encode(self, context, response):
        enc = self.tokenizer.encode(context, response)  # [CLS] 맥락 [SEP] 응답 [SEP]
        return (np.asarray([enc.ids], dtype=np.int64),
                np.asarray([enc.attention_mask], dtype=np.int64))

    def infer(self, ids, mask):
        return self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]


def run_child(args):
    name, model_dir = args.child_model.split("=", 1)
    model_dir = Path(model_dir)
    log = lambda msg: print(f"[{name} t{args.threads}] {msg}", flush=True)
    rows = load_data(args.split, None if args.no_eval else args.eval_limit)
    lat_rows = rows[:args.latency_samples]

    rss_before = current_rss_mb()
    t0 = time.perf_counter()
    model = Model(model_dir, args.threads, args.max_length)
    load_seconds = time.perf_counter() - t0
    rss_after = current_rss_mb()

    # 첫 추론 (cold): 앱을 켜고 처음 문장을 넣었을 때 체감 시간
    t0 = time.perf_counter()
    model.infer(*model.encode(lat_rows[0]["context"], lat_rows[0]["response"]))
    first_ms = (time.perf_counter() - t0) * 1000
    log(f"로드 {load_seconds:.2f}초, 첫 추론 {first_ms:.0f} ms")

    encoded = [model.encode(r["context"], r["response"]) for r in lat_rows]
    for i in range(args.warmup):
        model.infer(*encoded[i % len(encoded)])

    tok_ms, inf_ms = [], []
    for i in range(args.runs):
        row = lat_rows[i % len(lat_rows)]
        t0 = time.perf_counter()
        ids, mask = model.encode(row["context"], row["response"])
        t1 = time.perf_counter()
        model.infer(ids, mask)
        t2 = time.perf_counter()
        tok_ms.append((t1 - t0) * 1000)
        inf_ms.append((t2 - t1) * 1000)
    e2e_ms = [a + b for a, b in zip(tok_ms, inf_ms)]
    log(f"추론 지연시간 중앙값 {statistics.median(inf_ms):.1f} ms, "
        f"토크나이저 {statistics.median(tok_ms):.2f} ms")

    model_files = [model_dir / "model.onnx"] + sorted(model_dir.glob("model.onnx.data"))
    seq_lens = [int(ids.shape[1]) for ids, _ in encoded]
    result = {
        "model": name,
        "model_dir": os.path.relpath(model_dir, PROJECT_ROOT),
        "threads": args.threads,
        "file_size_mb": sum(p.stat().st_size for p in model_files) / 1024 ** 2,
        "load_seconds": load_seconds,
        "first_inference_ms": first_ms,
        "latency_ms": {"inference": stats_ms(inf_ms), "tokenizer": stats_ms(tok_ms),
                       "end_to_end": stats_ms(e2e_ms)},
        "latency_inputs": {"samples": len(encoded), "seq_len_mean": statistics.mean(seq_lens),
                           "seq_len_max": max(seq_lens)},
    }

    # 지속 부하: 앱을 계속 쓸 때 발열로 느려지는지 본다
    if args.sustained_seconds > 0:
        log(f"지속 부하 {args.sustained_seconds:.0f}초 시작")
        timeline, window, i = [], [], 0
        start = last = time.perf_counter()
        while (now := time.perf_counter()) - start < args.sustained_seconds:
            t0 = time.perf_counter()
            model.infer(*encoded[i % len(encoded)])
            window.append((time.perf_counter() - t0) * 1000)
            i += 1
            if time.perf_counter() - last >= args.sustained_window:
                last = time.perf_counter()
                timeline.append({"t_seconds": round(last - start, 1),
                                 "runs": len(window),
                                 "median_ms": statistics.median(window),
                                 "temperature_c": temperatures_c(),
                                 "cpu_mhz": cpu_freqs_mhz()})
                log(f"  {timeline[-1]['t_seconds']:>5.0f}초: 중앙값 "
                    f"{timeline[-1]['median_ms']:.1f} ms, 온도 {timeline[-1]['temperature_c']}")
                window = []
        if window and (not timeline or len(window) >= timeline[-1]["runs"] // 2):
            timeline.append({"t_seconds": round(time.perf_counter() - start, 1),
                             "runs": len(window), "median_ms": statistics.median(window),
                             "temperature_c": temperatures_c(), "cpu_mhz": cpu_freqs_mhz()})
        first_w, last_w = timeline[0]["median_ms"], timeline[-1]["median_ms"]
        result["sustained"] = {
            "seconds": args.sustained_seconds, "window_seconds": args.sustained_window,
            "total_runs": i,
            "first_window_median_ms": first_w, "last_window_median_ms": last_w,
            "slowdown_ratio": last_w / first_w,
            "timeline": timeline,
        }

    # 정확도: 앱과 같은 조건(batch 1, 패딩 없음)으로 문장마다 추론
    if not args.no_eval:
        logits = []
        t0 = time.perf_counter()
        for i, row in enumerate(rows, 1):
            logits.append(model.infer(*model.encode(row["context"], row["response"]))[0])
            if i % 200 == 0 or i == len(rows):
                log(f"평가 {i}/{len(rows)}")
        seconds = time.perf_counter() - t0
        logits = np.asarray(logits, dtype=np.float32)
        metrics = compute_metrics([r["label"] for r in rows], logits.argmax(1))
        result["eval"] = {"split": args.split, "samples": len(rows), **metrics,
                          "seconds": seconds, "samples_per_second": len(rows) / seconds}
        np.save(args.child_output.with_suffix(".npy"), logits)
        log(f"정확도 {metrics['accuracy']:.4f}, F1(S) {metrics['f1_sarcasm']:.4f}, "
            f"BA {metrics['balanced_accuracy']:.4f}")

    result["memory_mb"] = {
        "before_load": rss_before, "after_load": rss_after,
        "model_load_increase": (rss_after - rss_before
                                if rss_before is not None and rss_after is not None else None),
        "peak": peak_rss_mb(),
    }
    args.child_output.write_text(json.dumps(result), encoding="utf-8")


# ============================
# 전체 실행 (부모 프로세스)
# ============================

def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def pad(text, width):
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def print_summary(results):
    g = lambda *keys: (lambda r: _dig(r, keys))
    rows = [
        ("파일 크기 (MB)", g("file_size_mb"), ".1f"),
        ("모델 로드 (초)", g("load_seconds"), ".2f"),
        ("첫 추론 (ms)", g("first_inference_ms"), ".0f"),
        ("추론 중앙값 (ms)", g("latency_ms", "inference", "median"), ".1f"),
        ("추론 p90 (ms)", g("latency_ms", "inference", "p90"), ".1f"),
        ("토크나이저 중앙값 (ms)", g("latency_ms", "tokenizer", "median"), ".2f"),
        ("전체 중앙값 (ms)", g("latency_ms", "end_to_end", "median"), ".1f"),
        ("지속 부하 느려짐 (배)", g("sustained", "slowdown_ratio"), ".2f"),
        ("로드 후 메모리 증가 (MB)", g("memory_mb", "model_load_increase"), ".0f"),
        ("최대 메모리 (MB)", g("memory_mb", "peak"), ".0f"),
        ("정확도", g("eval", "accuracy"), ".4f"),
        ("F1(S)", g("eval", "f1_sarcasm"), ".4f"),
        ("BA", g("eval", "balanced_accuracy"), ".4f"),
        ("Macro-F1", g("eval", "macro_f1"), ".4f"),
        ("처리량 (개/초, batch 1)", g("eval", "samples_per_second"), ".2f"),
        ("기준 대비 예측 일치율", g("reference", "prediction_agreement"), ".4f"),
        ("기준 대비 최대 logit 차이", g("reference", "max_logit_diff"), ".2e"),
    ]
    names = [f"{r['model']}/t{r['threads']}" for r in results]
    width = max(16, *(len(n) + 2 for n in names))
    print("\n===== 결과 =====")
    print(pad("항목", 28) + "".join(f"{n:>{width}}" for n in names))
    for label, getter, spec in rows:
        print(pad(label, 28) + "".join(f"{fmt(getter(r), spec):>{width}}" for r in results))


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def run(args):
    models = []
    for spec in args.models:
        if "=" not in spec:
            sys.exit(f"--models 형식은 이름=폴더 입니다: {spec}")
        name, directory = spec.split("=", 1)
        directory = Path(directory).resolve()
        for f in ("model.onnx", "tokenizer.json"):
            path = directory / f
            if not path.exists():
                sys.exit(f"{name}: {path} 가 없습니다.")
            if f == "model.onnx" and path.stat().st_size < 10_000:
                sys.exit(f"{name}: {path} 가 너무 작습니다. git lfs pull 을 했는지 확인하세요.")
        models.append((name, directory))
    load_data(args.split, 1)  # 데이터 파일이 있는지 먼저 확인

    reference = None
    if args.reference:
        reference = dict(np.load(args.reference))
        print(f"기준 결과: {args.reference} ({', '.join(reference)})")

    device = device_info(args.device_name)
    print(f"기기: {device.get('android_model') or device['platform']} "
          f"({device['machine']}, 코어 {device['cpu_count']}), "
          f"온도 {device['temperature_c']}, 배터리 {device['battery']}")

    results, all_logits = [], {}
    jobs = [(name, d, t) for t in args.threads for name, d in models]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, directory, threads) in enumerate(jobs):
            if i > 0 and args.cooldown > 0:
                print(f"\n{args.cooldown:.0f}초 대기 (발열 완화)...", flush=True)
                time.sleep(args.cooldown)
            print(f"\n===== {name} (스레드 {threads}) =====", flush=True)
            out = Path(tmp) / f"{i}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "child",
                   "--child-model", f"{name}={directory}", "--child-output", str(out),
                   "--threads", str(threads), "--split", args.split,
                   "--max-length", str(args.max_length),
                   "--latency-samples", str(args.latency_samples),
                   "--warmup", str(args.warmup), "--runs", str(args.runs),
                   "--sustained-seconds", str(args.sustained_seconds),
                   "--sustained-window", str(args.sustained_window)]
            if args.no_eval:
                cmd.append("--no-eval")
            if args.eval_limit:
                cmd += ["--eval-limit", str(args.eval_limit)]
            subprocess.run(cmd, check=True)
            result = json.loads(out.read_text(encoding="utf-8"))
            result["temperature_after_c"] = temperatures_c()
            if out.with_suffix(".npy").exists():
                logits = np.load(out.with_suffix(".npy"))
                all_logits[f"{name}__t{threads}"] = logits
                if reference is not None:
                    ref = _match_reference(reference, name, threads)
                    if ref is None:
                        print(f"  기준 결과에 {name} 이(가) 없어 비교를 건너뜁니다.")
                    elif len(ref) != len(logits):
                        print(f"  기준 결과의 샘플 수({len(ref)})가 달라 비교를 건너뜁니다.")
                    else:
                        result["reference"] = {
                            "prediction_agreement": float((ref.argmax(1) == logits.argmax(1)).mean()),
                            "changed_predictions": int((ref.argmax(1) != logits.argmax(1)).sum()),
                            "max_logit_diff": float(np.abs(ref - logits).max()),
                            "mean_logit_diff": float(np.abs(ref - logits).mean()),
                        }
            results.append(result)

    print_summary(results)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                     if k not in ("func",)},
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device["name"])
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{safe}_{args.split}"
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {json_path}")
    if all_logits:
        np.savez_compressed(args.output_dir / f"{stem}.npz", **all_logits)
        print(f"logits 저장: {args.output_dir / (stem + '.npz')}  (다른 기기에서 --reference로 사용)")


def _match_reference(reference, name, threads):
    """같은 모델·스레드 결과를 우선 쓰고, 없으면 같은 모델의 아무 스레드 결과를 쓴다."""
    if f"{name}__t{threads}" in reference:
        return reference[f"{name}__t{threads}"]
    for key, value in reference.items():
        if key.split("__t")[0] == name:
            return value
    return None


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="(PC) KoCoSa를 data/*.jsonl로 저장한다. datasets 필요")
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.set_defaults(func=prepare)

    def common(p):
        p.add_argument("--split", choices=("validation", "test"), default="test")
        p.add_argument("--max-length", type=int, default=256)
        p.add_argument("--latency-samples", type=int, default=32,
                       help="지연시간 측정에 돌려 쓸 문장 수 (데이터 앞에서부터)")
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--runs", type=int, default=100)
        p.add_argument("--no-eval", action="store_true", help="정확도 평가 생략 (속도·메모리만)")
        p.add_argument("--eval-limit", type=int, default=None, help="평가에 쓸 최대 문장 수")
        p.add_argument("--sustained-seconds", type=float, default=0,
                       help="지속 부하 측정 시간(초). 0이면 생략. 폰에서는 120 이상 권장")
        p.add_argument("--sustained-window", type=float, default=10,
                       help="지속 부하 기록 간격(초)")

    p = sub.add_parser("run", help="모델들을 측정한다 (PC, 폰 공통)")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help="이름=폴더 목록. 폴더에 model.onnx와 tokenizer.json이 있어야 한다")
    p.add_argument("--threads", type=int, nargs="+", default=[4],
                   help="스레드 수. 여러 개 주면 모두 측정 (예: 1 2 4 8)")
    p.add_argument("--cooldown", type=float, default=30, help="측정 사이 대기(초)")
    p.add_argument("--reference", type=Path, help="비교 기준 .npz (보통 PC에서 만든 결과)")
    p.add_argument("--device-name", help="결과 파일에 남길 기기 이름 (예: galaxy-s24)")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "mobile")
    common(p)
    p.set_defaults(func=run)

    p = sub.add_parser("child")
    p.add_argument("--child-model", required=True)
    p.add_argument("--child-output", type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    common(p)
    p.set_defaults(func=run_child)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
