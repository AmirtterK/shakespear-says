"""Export the fine-tuned checkpoint to the exact folder layout transformers.js (v2) loads,
then int8-quantize it and sanity-check the result.

    python scripts/export_onnx.py

Input : build/shakespeare-gpt2/            (from train_gpt2.py)
Output: build/shakespeare-gpt2-onnx/       (drop-in replacement for src/lib/shakespeare-gpt2-onnx)
        build/shakespeare-gpt2-onnx.zip

    shakespeare-gpt2-onnx/
      config.json  generation_config.json  tokenizer.json  tokenizer_config.json
      vocab.json   merges.txt              special_tokens_map.json
      onnx/decoder_model_merged_quantized.onnx      <- what the app loads (quantized: true)
      onnx/decoder_model_merged.onnx                <- only with INCLUDE_FP32=1

Environment overrides: MODEL_DIR, ONNX_DIR, INCLUDE_FP32=1, PER_CHANNEL=0, REDUCE_RANGE=0,
QUANT_OPS="MatMul" (comma separated; add Gather to also quantize embeddings -- smaller, lower quality).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "build/shakespeare-gpt2"))
ONNX_DIR = Path(os.environ.get("ONNX_DIR", "build/shakespeare-gpt2-onnx"))
RAW_DIR = ONNX_DIR.parent / "onnx-raw"
CSV_PATH = Path(os.environ.get("CSV_PATH", "data/raw/Shakespeare_data.csv"))
INCLUDE_FP32 = os.environ.get("INCLUDE_FP32") == "1"
PER_CHANNEL = os.environ.get("PER_CHANNEL", "1") != "0"
REDUCE_RANGE = os.environ.get("REDUCE_RANGE", "1") != "0"
QUANT_OPS = [op.strip() for op in os.environ.get("QUANT_OPS", "MatMul").split(",") if op.strip()]

# @xenova/transformers v2 ships onnxruntime 1.14, which reads IR version <= 8 and opset <= 18.
MAX_IR_VERSION = 8
MAX_OPSET = 18

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


# optimum 1.x uses torch's classic (TorchScript) ONNX exporter. Newer torch versions default to
# the dynamo exporter, which produces different graphs and breaks the merged decoder, so we
# force dynamo=False before optimum is imported.
EXPORT_WRAPPER = (
    "import functools, inspect, torch\n"
    "_orig = torch.onnx.export\n"
    "if 'dynamo' in inspect.signature(_orig).parameters:\n"
    "    @functools.wraps(_orig)\n"
    "    def _export(*a, **k):\n"
    "        k.setdefault('dynamo', False)\n"
    "        return _orig(*a, **k)\n"
    "    torch.onnx.export = _export\n"
    "from optimum.exporters.onnx.__main__ import main\n"
    "main()\n"
)


def run_export() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    command = [
        sys.executable,
        "-c",
        EXPORT_WRAPPER,
        "--model",
        str(MODEL_DIR),
        "--task",
        "text-generation-with-past",
        str(RAW_DIR),
    ]
    print("Exporting", MODEL_DIR, "->", RAW_DIR)
    subprocess.run(command, check=True)

    merged = RAW_DIR / "decoder_model_merged.onnx"
    if not merged.exists():
        print("No merged decoder produced -- merging decoder + decoder_with_past ourselves")
        from optimum.onnx.graph_transformations import merge_decoders

        merged_model = merge_decoders(RAW_DIR / "decoder_model.onnx", RAW_DIR / "decoder_with_past_model.onnx")
        onnx.save(merged_model, str(merged))


def fix_model_for_old_runtime(path: Path) -> None:
    model = onnx.load(str(path))
    default_opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
    print(f"{path.name}: ir_version={model.ir_version} opset={default_opset}")
    if default_opset > MAX_OPSET:
        raise RuntimeError(
            f"{path.name} uses opset {default_opset} but onnxruntime in transformers.js v2 supports <= {MAX_OPSET}. "
            "Re-run with the pinned optimum/transformers versions from the Colab notebook."
        )
    if model.ir_version > MAX_IR_VERSION:
        model.ir_version = MAX_IR_VERSION
        onnx.save(model, str(path))
        print(f"  lowered ir_version to {MAX_IR_VERSION}")


def quantize(merged: Path, quantized: Path) -> None:
    print(f"Quantizing (ops={QUANT_OPS}, per_channel={PER_CHANNEL}, reduce_range={REDUCE_RANGE}) ...")
    try:
        from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import QuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(RAW_DIR), file_name=merged.name)
        qconfig = QuantizationConfig(
            is_static=False,
            format=QuantFormat.QOperator,
            mode=QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QUInt8,
            weights_dtype=QuantType.QUInt8,
            per_channel=PER_CHANNEL,
            reduce_range=REDUCE_RANGE,
            operators_to_quantize=QUANT_OPS,
        )
        quantizer.quantize(save_dir=str(RAW_DIR), quantization_config=qconfig)
        produced = RAW_DIR / f"{merged.stem}_quantized.onnx"
        if produced != quantized:
            shutil.move(str(produced), str(quantized))
    except Exception as error:  # noqa: BLE001
        print(f"ORTQuantizer path failed ({error!r}); falling back to quantize_dynamic")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(merged),
            model_output=str(quantized),
            weight_type=QuantType.QUInt8,
            per_channel=PER_CHANNEL,
            reduce_range=REDUCE_RANGE,
            op_types_to_quantize=QUANT_OPS,
        )


# ----------------------------------------------------------------------------
# Verification: perplexity + greedy continuation from the actual ONNX files
# ----------------------------------------------------------------------------


class OnnxDecoder:
    def __init__(self, path: Path, config: dict):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.head_dim = config["n_embd"] // config["n_head"]

    def _empty_past(self) -> dict:
        feeds = {}
        for layer in range(self.n_layer):
            for kind in ("key", "value"):
                feeds[f"past_key_values.{layer}.{kind}"] = np.zeros((1, self.n_head, 0, self.head_dim), dtype=np.float32)
        return feeds

    def _run(self, ids: np.ndarray, past: dict | None, past_len: int):
        feeds = {"input_ids": ids.astype(np.int64)}
        total = past_len + ids.shape[1]
        feeds["attention_mask"] = np.ones((1, total), dtype=np.int64)
        if "position_ids" in self.input_names:
            feeds["position_ids"] = np.arange(past_len, total, dtype=np.int64)[None, :]
        if "use_cache_branch" in self.input_names:
            feeds["use_cache_branch"] = np.array([past is not None], dtype=bool)
        feeds.update(past if past is not None else self._empty_past())
        names = [o.name for o in self.session.get_outputs()]
        outputs = dict(zip(names, self.session.run(None, feeds)))
        new_past = {n.replace("present.", "past_key_values."): v for n, v in outputs.items() if n.startswith("present.")}
        return outputs["logits"], new_past

    def nll(self, ids: list[int], skip: int) -> tuple[float, int]:
        logits, _ = self._run(np.array([ids]), None, 0)
        logits = logits[0].astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        total, count = 0.0, 0
        for position in range(max(skip, 1), len(ids)):
            total -= log_probs[position - 1, ids[position]]
            count += 1
        return total, count

    def greedy(self, ids: list[int], steps: int, eos: int) -> list[int]:
        logits, past = self._run(np.array([ids]), None, 0)
        out: list[int] = []
        past_len = len(ids)
        for _ in range(steps):
            token = int(logits[0, -1].argmax())
            if token == eos:
                break
            out.append(token)
            logits, past = self._run(np.array([[token]]), past, past_len)
            past_len += 1
        return out


def verify(fp32: Path | None, quantized: Path) -> None:
    from transformers import GPT2TokenizerFast

    config = json.loads((RAW_DIR / "config.json").read_text(encoding="utf-8"))
    tokenizer = GPT2TokenizerFast.from_pretrained(str(MODEL_DIR))
    eos = tokenizer.eos_token_id

    all_samples = prepare_data.build_samples(CSV_PATH)
    samples = all_samples[:: max(1, len(all_samples) // 150)][:150]

    models = {"quantized": OnnxDecoder(quantized, config)}
    if fp32 is not None:
        models["fp32"] = OnnxDecoder(fp32, config)

    print("\n=== perplexity on 150 dataset samples (lower is better) ===")
    perplexities = {}
    for name, decoder in models.items():
        total, count = 0.0, 0
        for sample in samples:
            header = sample.split("\n", 1)[0] + "\n"
            ids = tokenizer(sample).input_ids + [eos]
            if len(ids) > 160:
                continue
            sample_total, sample_count = decoder.nll(ids, skip=len(tokenizer(header).input_ids))
            total += sample_total
            count += sample_count
        perplexities[name] = math.exp(total / max(count, 1))
        print(f"{name:10s} perplexity = {perplexities[name]:.2f}")
    if "fp32" in perplexities and perplexities["quantized"] > perplexities["fp32"] * 1.15:
        print("WARNING: quantization costs >15% perplexity. Re-run with REDUCE_RANGE=0 or PER_CHANNEL=0, or ship fp32.")

    print("\n=== greedy continuations from the ONNX files ===")
    for speaker in ["HAMLET", "PORTIA", "KING LEAR"]:
        prompt_ids = tokenizer(f"{speaker}:\n").input_ids
        for name, decoder in models.items():
            text = tokenizer.decode(decoder.greedy(prompt_ids, 50, eos))
            print(f"[{name}] {speaker}: {text!r}")


def package() -> Path:
    if ONNX_DIR.exists():
        shutil.rmtree(ONNX_DIR)
    (ONNX_DIR / "onnx").mkdir(parents=True)

    for name in ["config.json", "generation_config.json"]:
        shutil.copy(RAW_DIR / name, ONNX_DIR / name)
    for name in TOKENIZER_FILES:
        source = RAW_DIR / name if (RAW_DIR / name).exists() else MODEL_DIR / name
        if source.exists():
            shutil.copy(source, ONNX_DIR / name)

    shutil.copy(RAW_DIR / "decoder_model_merged_quantized.onnx", ONNX_DIR / "onnx" / "decoder_model_merged_quantized.onnx")
    if INCLUDE_FP32:
        shutil.copy(RAW_DIR / "decoder_model_merged.onnx", ONNX_DIR / "onnx" / "decoder_model_merged.onnx")

    archive = shutil.make_archive(str(ONNX_DIR), "zip", root_dir=ONNX_DIR.parent, base_dir=ONNX_DIR.name)
    return Path(archive)


def main() -> None:
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit(f"{MODEL_DIR} not found -- run scripts/train_gpt2.py first.")

    run_export()
    merged = RAW_DIR / "decoder_model_merged.onnx"
    quantized = RAW_DIR / "decoder_model_merged_quantized.onnx"

    fix_model_for_old_runtime(merged)
    quantize(merged, quantized)
    fix_model_for_old_runtime(quantized)

    for path in (merged, quantized):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")

    try:
        verify(merged, quantized)
    except Exception as error:  # noqa: BLE001
        print(f"\nVerification step failed ({error!r}). The export itself finished; test it in the app.")

    archive = package()
    print(f"\nDone. Download {archive} and unzip it into src/lib/ (replace the old shakespeare-gpt2-onnx folder).")


if __name__ == "__main__":
    main()
