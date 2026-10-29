"""Fine-tune GPT-2 (124M) on speaker-formatted Shakespeare samples.

Runs locally or on Colab (use a GPU runtime -- a free T4 takes ~10-20 min).

    python scripts/train_gpt2.py

Environment overrides: CSV_PATH, OUT_DIR, BASE_MODEL, EPOCHS, BATCH_SIZE, LEARNING_RATE.

Output: build/shakespeare-gpt2/  (a normal Hugging Face checkpoint, input to export_onnx.py)

Why this differs from the old script:
  * trains on the SAME format the API prompts with ("SPEAKER:\\nline\\nline<eos>")
  * no stage directions / ACT / SCENE / quote marks in the data
  * one sample per short speech chunk, with a real EOS at the end so the model learns
    to stop (the stock LM collator masks EOS when pad == eos, which breaks this)
  * held-out validation set, best checkpoint kept by validation loss
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data  # noqa: E402

CSV_PATH = Path(os.environ.get("CSV_PATH", "data/raw/Shakespeare_data.csv"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "build/shakespeare-gpt2"))
CHECKPOINT_DIR = OUT_DIR.parent / "checkpoints"

BASE_MODEL = os.environ.get("BASE_MODEL", "gpt2")  # any GPT-2 style checkpoint name or local folder
MAX_TOKENS = 160  # samples longer than this are dropped (never truncated -- EOS must survive)
EPOCHS = float(os.environ.get("EPOCHS", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 5e-5))
VAL_FRACTION = 0.05
SEED = 42


class SpeechDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def encode_samples(samples: list[str], tokenizer: GPT2TokenizerFast) -> tuple[list[dict], int]:
    items: list[dict] = []
    dropped = 0
    for sample in samples:
        header = sample.split("\n", 1)[0] + "\n"  # "HAMLET:\n"
        ids = tokenizer(sample).input_ids + [tokenizer.eos_token_id]
        if len(ids) > MAX_TOKENS:
            dropped += 1
            continue
        items.append({"ids": ids, "prefix": len(tokenizer(header).input_ids)})
    return items, dropped


def split_blocks(items: list[dict], block: int = 50) -> tuple[list[dict], list[dict]]:
    """Split by contiguous blocks so chunks of one speech don't leak across train/val."""
    blocks = [items[i : i + block] for i in range(0, len(items), block)]
    random.Random(SEED).shuffle(blocks)
    n_val = max(1, int(len(blocks) * VAL_FRACTION))
    val = [x for b in blocks[:n_val] for x in b]
    train = [x for b in blocks[n_val:] for x in b]
    return train, val


def make_collator(eos_id: int):
    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["ids"]) for item in batch)
        input_ids = torch.full((len(batch), longest), eos_id, dtype=torch.long)
        attention = torch.zeros((len(batch), longest), dtype=torch.long)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for row, item in enumerate(batch):
            ids = torch.tensor(item["ids"], dtype=torch.long)
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1
            labels[row, : len(ids)] = ids
            labels[row, : item["prefix"]] = -100  # don't spend capacity on the speaker header
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    return collate


@torch.no_grad()
def show_samples(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    device = next(model.parameters()).device
    model.eval()
    for speaker in ["HAMLET", "PORTIA", "KING LEAR", "FALSTAFF", "JULIET", "IAGO"]:
        prompt = tokenizer(f"{speaker}:\n", return_tensors="pt").to(device)
        output = model.generate(
            **prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("-" * 40)


def main() -> None:
    set_seed(SEED)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: no GPU found -- training will be very slow. On Colab: Runtime > Change runtime type > T4 GPU.")

    print(f"Building samples from {CSV_PATH} ...")
    samples = prepare_data.build_samples(CSV_PATH)
    print(f"{len(samples)} samples")

    tokenizer = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    items, dropped = encode_samples(samples, tokenizer)
    train_items, val_items = split_blocks(items)
    print(f"train={len(train_items)} val={len(val_items)} dropped_too_long={dropped}")

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SpeechDataset(train_items),
        eval_dataset=SpeechDataset(val_items),
        data_collator=make_collator(tokenizer.eos_token_id),
    )

    trainer.train()
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    perplexity = math.exp(eval_loss)
    print(f"Best validation loss {eval_loss:.4f} (perplexity {perplexity:.2f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True  # the old checkpoint was saved with use_cache=false
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "eval_loss": eval_loss,
                "perplexity": perplexity,
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "epochs": EPOCHS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model to {OUT_DIR}")

    print("\n=== sample generations (check these look like verse before exporting) ===\n")
    show_samples(model, tokenizer)


if __name__ == "__main__":
    main()
