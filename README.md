# Korean Sarcasm Detection Model

This repository contains a fine-tuned Korean RoBERTa sequence-classification model. The ONNX export uses the local model files and can be run without downloading a checkpoint.

Project goals, training results, and the roadmap (ONNX baseline → pruning → quantization) are described in [docs/PROJECT.md](docs/PROJECT.md).

## Folder layout

```
model/      Fine-tuned PyTorch checkpoint (safetensors, config, tokenizer)
onnx/       ONNX export of the checkpoint (model.onnx, config, tokenizer)
scripts/    train.py (fine-tuning), export_onnx.py (ONNX export and check),
            benchmark.py (PyTorch vs ONNX speed/memory/accuracy),
            baseline_eval.py (before vs after fine-tuning)
results/    result.txt (fine-tuning log), benchmark/, baseline_eval.json
docs/       Project documentation
```

## Export to ONNX

Install the export and inference dependencies:

```powershell
python -m pip install -r requirements-onnx.txt
```

Export the checkpoint in `model\` to `onnx\model.onnx`:

```powershell
python scripts\export_onnx.py
```

To select other paths or a different maximum input length:

```powershell
python scripts\export_onnx.py --model-dir model --output-dir onnx --max-length 256
```

The exported graph accepts dynamic batch and sequence lengths and returns classification logits. The script checks the graph and compares ONNX Runtime CPU output against PyTorch before reporting success. It also saves the tokenizer and model configuration alongside the ONNX file.

The inference session and NumPy input pattern follows the ONNX Runtime Python API sample in [onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/api).

## Labels

- `0`: Non-Sarcasm
- `1`: Sarcasm
