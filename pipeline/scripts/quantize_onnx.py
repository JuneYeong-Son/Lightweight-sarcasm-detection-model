"""4단계: FP32 ONNX에 ORTQuantizer + arm64(is_static=False) 동적 INT8 양자화를 적용한다.

동적 양자화: 가중치는 미리 int8로 저장하고, 활성값은 추론 때마다 범위를 계산해 int8로 바꾼다.
보정(calibration) 데이터가 필요 없다. arm64 설정은 안드로이드(ARM) 기기 실행을 목표로 한다.

    .venv/Scripts/python pipeline/scripts/quantize_onnx.py
    .venv/Scripts/python pipeline/scripts/quantize_onnx.py --per-channel
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PIPELINE_DIR.parent


def use_ascii_tempdir() -> None:
    """ONNX의 C++ shape 추론은 Windows에서 한글 등 비ASCII 경로를 열지 못한다.
    %TEMP%가 비ASCII이면 ASCII 경로(기본 C:\\Users\\Public\\ort_tmp, 환경 변수 ORT_QUANT_TMP로 변경)로 바꾼다."""
    if tempfile.gettempdir().isascii():
        return
    target = Path(os.environ.get("ORT_QUANT_TMP", r"C:\Users\Public\ort_tmp"))
    if not str(target).isascii():
        raise SystemExit(f"ORT_QUANT_TMP는 ASCII 경로여야 합니다: {target}")
    target.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(target)
    os.environ["TMP"] = os.environ["TEMP"] = str(target)
    print(f"임시 폴더를 ASCII 경로로 변경: {target}", flush=True)


def run(onnx_path: Path, input_ids, attention_mask):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=PIPELINE_DIR / "3_onnx_fp32",
                        help="FP32 model.onnx 폴더 (기본: pipeline/3_onnx_fp32/)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "onnx-pruned-int8",
                        help="양자화 결과 폴더 (기본: onnx-pruned-int8/)")
    parser.add_argument("--per-channel", action="store_true",
                        help="출력 채널별 scale 사용 (정확도↑, 크기 약간↑)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    use_ascii_tempdir()

    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=args.per_channel)
    quantizer = ORTQuantizer.from_pretrained(input_dir, file_name="model.onnx")
    print("동적 INT8 양자화 (arm64)...", flush=True)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    # Optimum은 model_quantized.onnx로 저장한다. 다른 폴더와 맞추려고 model.onnx로 이름을 바꾼다.
    quantized = output_dir / "model_quantized.onnx"
    dst = output_dir / "model.onnx"
    if quantized.exists():
        quantized.replace(dst)
    tokenizer = AutoTokenizer.from_pretrained(input_dir, local_files_only=True)
    tokenizer.save_pretrained(output_dir)

    # 검증: FP32와 같은 예측인지 확인
    enc = tokenizer(
        ["오늘 날씨가 참 좋네요.", "회의가 또 연기됐대."],
        ["비가 쏟아지는데 정말 완벽한 날씨야.", "그래, 덕분에 일이 아주 잘 풀리겠네."],
        truncation=True, max_length=256, padding=True, return_tensors="np",
    )
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    src = input_dir / "model.onnx"
    fp32 = run(src, ids, mask)
    int8 = run(dst, ids, mask)

    mb = 1024 ** 2
    print(f"\n양자화 설정: {qconfig}")
    print(f"파일 크기: {src.stat().st_size / mb:.1f} MB -> {dst.stat().st_size / mb:.1f} MB")
    print(f"logit 최대 차이: {np.abs(fp32 - int8).max():.4f}")
    print(f"예측 fp32 {fp32.argmax(1).tolist()} / int8 {int8.argmax(1).tolist()}")
    print(f"저장: {dst}")


if __name__ == "__main__":
    main()
