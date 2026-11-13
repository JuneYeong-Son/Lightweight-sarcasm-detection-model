"""ONNX fp32 모델을 동적 INT8로 양자화한다.

동적 양자화(dynamic quantization):
- 가중치(MatMul/Gemm 등)는 미리 int8로 바꿔서 저장한다.
- 활성값은 추론할 때마다 범위를 계산해서 int8로 바꾼다. 그래서 보정(calibration) 데이터가 필요 없다.
- Transformer(BERT/RoBERTa) 계열에서 ONNX Runtime이 권장하는 기본 방법이다.

    python scripts/quantize_onnx.py
    python scripts/quantize_onnx.py --per-channel
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIDE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def use_ascii_tempdir() -> None:
    """ONNX의 C++ shape 추론은 Windows에서 한글 등 비ASCII 경로의 파일을 열지 못한다.
    양자화 함수가 내부적으로 임시 폴더(%TEMP%, 예: C:\\Users\\손\\AppData\\...)에 파일을 쓰므로,
    임시 폴더 경로가 ASCII가 아니면 ASCII 경로(기본 C:\\Users\\Public\\ort_tmp)로 바꾼다.
    다른 위치를 쓰려면 환경 변수 ORT_QUANT_TMP로 지정한다."""
    current = tempfile.gettempdir()
    if current.isascii():
        return
    target = Path(os.environ.get("ORT_QUANT_TMP", r"C:\Users\Public\ort_tmp"))
    if not str(target).isascii():
        raise SystemExit(f"ORT_QUANT_TMP는 ASCII 경로여야 합니다: {target}")
    target.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(target)
    print(f"임시 폴더를 ASCII 경로로 변경: {target}", flush=True)


def run(onnx_path: Path, input_ids, attention_mask):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "onnx",
                        help="fp32 model.onnx가 있는 폴더 (기본: onnx/)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "onnx-int8",
                        help="양자화 결과 폴더 (기본: onnx-int8/)")
    parser.add_argument("--per-channel", action="store_true",
                        help="출력 채널별로 scale을 따로 둔다 (정확도↑, 크기 약간↑)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="양자화 전 shape 추론/그래프 정리 단계 생략")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    src = input_dir / "model.onnx"
    dst = output_dir / "model.onnx"

    use_ascii_tempdir()

    # 1) 전처리: shape 추론. 양자화할 노드를 더 잘 찾게 해 준다.
    model = onnx.load(str(src))
    if not args.skip_preprocess:
        print("전처리 (shape 추론)...", flush=True)
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)

    # 2) 동적 INT8 양자화
    print("동적 INT8 양자화...", flush=True)
    quantize_dynamic(
        model_input=model,
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )
    onnx.checker.check_model(str(dst))

    for name in SIDE_FILES:
        if (input_dir / name).exists():
            shutil.copy2(input_dir / name, output_dir / name)

    # 3) 간단한 검증: fp32와 같은 예측인지 확인
    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    enc = tokenizer(
        ["오늘 날씨가 참 좋네요.", "회의가 또 연기됐대."],
        ["비가 쏟아지는데 정말 완벽한 날씨야.", "그래, 덕분에 일이 아주 잘 풀리겠네."],
        truncation=True, max_length=256, padding=True, return_tensors="np",
    )
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    fp32 = run(src, ids, mask)
    int8 = run(dst, ids, mask)

    mb = 1024 ** 2
    print(f"\n파일 크기: {src.stat().st_size / mb:.1f} MB -> {dst.stat().st_size / mb:.1f} MB")
    print(f"logit 최대 차이: {np.abs(fp32 - int8).max():.4f}")
    print(f"예측 fp32 {fp32.argmax(1).tolist()} / int8 {int8.argmax(1).tolist()}")
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
