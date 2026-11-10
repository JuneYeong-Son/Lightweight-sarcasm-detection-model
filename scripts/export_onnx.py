import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ClassificationLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the local sarcasm classifier to ONNX and verify it with ONNX Runtime."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "model",
        help="Directory containing the fine-tuned model and tokenizer (default: model/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "onnx",
        help="Directory for model.onnx and tokenizer files (default: onnx/).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length for the verification example.",
    )
    args = parser.parse_args()

    if args.max_length < 2:
        parser.error("--max-length must be at least 2")

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    model.to("cpu")
    export_model = ClassificationLogits(model)
    export_model.eval()

    encoded = tokenizer(
        "오늘 날씨가 참 좋네요.",
        "비가 쏟아지는데 정말 완벽한 날씨야.",
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    torch.onnx.export(
        export_model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    verification_inputs = tokenizer(
        [
            "날씨가 좋다.",
            "하루 종일 비가 오고 바람까지 불어서 밖에 나가기 정말 힘들었어.",
        ],
        [
            "응.",
            "이보다 더 완벽한 날씨가 있을 수가 없네.",
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    verification_ids = verification_inputs["input_ids"]
    verification_mask = verification_inputs["attention_mask"]
    with torch.inference_mode():
        expected_logits = export_model(
            verification_ids,
            verification_mask,
        ).numpy()
    actual_logits = session.run(
        ["logits"],
        {
            "input_ids": verification_ids.numpy(),
            "attention_mask": verification_mask.numpy(),
        },
    )[0]
    np.testing.assert_allclose(
        actual_logits,
        expected_logits,
        rtol=1e-3,
        atol=1e-4,
    )

    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"ONNX model exported and verified: {onnx_path}")
    print(
        f"Verified input shape: {tuple(verification_ids.shape)}; "
        f"output shape: {actual_logits.shape}"
    )


if __name__ == "__main__":
    main()
