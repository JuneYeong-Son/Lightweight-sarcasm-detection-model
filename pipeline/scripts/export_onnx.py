"""3단계: 프루닝(0.6) + 회복학습이 끝난 PyTorch 모델을 FP32 ONNX로 변환한다.

Optimum의 ORTModelForSequenceClassification(export=True)으로 변환하므로,
4단계의 ORTQuantizer가 그대로 읽을 수 있는 폴더(model.onnx + config + tokenizer)가 만들어진다.

    .venv/Scripts/python pipeline/scripts/export_onnx.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PIPELINE_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=PIPELINE_DIR / "2_pruning" / "models",
                        help="프루닝된 PyTorch 모델 폴더 (기본: pipeline/2_pruning/models/)")
    parser.add_argument("--tokenizer-dir", type=Path, default=PIPELINE_DIR / "1_finetuned_model",
                        help="tokenizer.json이 있는 폴더 (기본: pipeline/1_finetuned_model/)")
    parser.add_argument("--output-dir", type=Path, default=PIPELINE_DIR / "3_onnx_fp32",
                        help="FP32 ONNX 저장 폴더 (기본: pipeline/3_onnx_fp32/)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 폴더에는 토크나이저가 없으므로 토크나이저는 루트에서 따로 읽는다.
    model_dir = args.model_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir.resolve(), local_files_only=True)
    torch_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).eval()

    print("FP32 ONNX 변환...", flush=True)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, export=True, local_files_only=True
    )
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 검증: PyTorch와 ONNX의 logit 비교
    enc = tokenizer(
        ["오늘 날씨가 참 좋네요.", "회의가 또 연기됐대."],
        ["비가 쏟아지는데 정말 완벽한 날씨야.", "그래, 덕분에 일이 아주 잘 풀리겠네."],
        truncation=True, max_length=256, padding=True, return_tensors="pt",
    )
    enc.pop("token_type_ids", None)
    with torch.inference_mode():
        expected = torch_model(**enc).logits.numpy()
    actual = ORTModelForSequenceClassification.from_pretrained(output_dir)(**enc).logits
    actual = actual.numpy() if hasattr(actual, "numpy") else np.asarray(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)

    print(f"logit 최대 차이 (PyTorch vs ONNX): {np.abs(actual - expected).max():.2e}")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
