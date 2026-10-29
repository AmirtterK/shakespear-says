"""Build clean, speaker-formatted training samples from data/raw/Shakespeare_data.csv.

Every sample looks like this (and the API route prompts the model in the same way):

    HAMLET:
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer

Rules:
  * rows with no Player (act/scene headers, "Enter ..." directions) are dropped
  * lines are grouped into speeches via (Play, PlayerLinenumber, Player) -- in this
    dataset PlayerLinenumber is really a per-play speech counter
  * inline directions like [Aside] or (to Hamlet) are stripped
  * long speeches are cut into chunks of at most MAX_LINES lines, each chunk
    gets the speaker header again, so the model learns short, self-contained output

Usage:
    python scripts/prepare_data.py            # writes data/processed/samples.jsonl + stats
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

CSV_PATH = Path("data/raw/Shakespeare_data.csv")
OUT_PATH = Path("data/processed/samples.jsonl")

MAX_LINES = 4  # lines per training sample
MIN_CHARS = 40  # drop samples shorter than this (after cleaning)
MIN_SPEAKER_LINES = 20  # drop speakers that barely speak (noise like "All", "Both")

_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_SPACES = re.compile(r"\s+")


def clean_line(text: str) -> str:
    text = _BRACKETS.sub(" ", text or "")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = text.strip().strip('"').strip()
    text = _SPACES.sub(" ", text)
    return text


def speaker_label(player: str) -> str:
    # Upper-case labels, e.g. "Clown" -> "CLOWN", so the speaker header is consistent.
    return _SPACES.sub(" ", player.strip()).upper()


def read_rows(csv_path: Path = CSV_PATH) -> list[dict]:
    with open(csv_path, encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def build_samples(csv_path: Path = CSV_PATH) -> list[str]:
    rows = [row for row in read_rows(csv_path) if (row.get("Player") or "").strip()]

    speaker_lines = Counter(speaker_label(row["Player"]) for row in rows)
    samples: list[str] = []

    def speech_key(row: dict) -> tuple[str, str, str]:
        return (row.get("Play", ""), row.get("PlayerLinenumber", ""), row["Player"])

    for _, group in groupby(rows, key=speech_key):
        group = list(group)
        speaker = speaker_label(group[0]["Player"])
        if speaker_lines[speaker] < MIN_SPEAKER_LINES:
            continue

        lines = [clean_line(row.get("PlayerLine", "")) for row in group]
        lines = [line for line in lines if len(line) > 1 and re.search(r"[A-Za-z]", line)]
        if not lines:
            continue

        for start in range(0, len(lines), MAX_LINES):
            chunk = lines[start : start + MAX_LINES]
            body = "\n".join(chunk)
            if len(body) < MIN_CHARS:
                continue
            samples.append(f"{speaker}:\n{body}")

    return samples


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    samples = build_samples(csv_path)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({"text": sample}, ensure_ascii=False) + "\n")

    speakers = Counter(sample.split(":\n", 1)[0] for sample in samples)
    print(f"Wrote {len(samples)} samples to {OUT_PATH}")
    print("Top speakers:", ", ".join(f"{name} ({count})" for name, count in speakers.most_common(12)))
    print("\n--- example samples ---")
    for sample in samples[:3] + samples[len(samples) // 2 : len(samples) // 2 + 2]:
        print(sample)
        print("---")


if __name__ == "__main__":
    main()
