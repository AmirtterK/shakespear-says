import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ZIP_PATH = Path("shakespear.zip")
MODEL_PATH = Path("src/lib/shakespeare-model.json")
ORDER = 2
MAX_CHOICES_PER_STATE = 14
MAX_STATES_PER_SEED = 100
MAX_RETRIEVAL_DOCS = 4500
LINES_PER_DOC = 4

STAGE_PATTERNS = (
    re.compile(r"^act\b", re.I),
    re.compile(r"^scene\b", re.I),
    re.compile(r"^enter\b", re.I),
    re.compile(r"^exit\b", re.I),
    re.compile(r"^exeunt\b", re.I),
    re.compile(r"^re-enter\b", re.I),
    re.compile(r"^alarum\b", re.I),
    re.compile(r"^flourish\b", re.I),
    re.compile(r"^aside\b", re.I),
    re.compile(r"^prologue\b", re.I),
    re.compile(r"^\["),
    re.compile(r"^\("),
)

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "for",
    "this",
    "you",
    "your",
    "are",
    "but",
    "not",
    "his",
    "her",
    "him",
    "our",
    "from",
    "have",
    "what",
    "when",
    "where",
    "will",
    "shall",
    "would",
    "could",
    "should",
    "there",
    "then",
    "than",
    "they",
    "them",
    "their",
    "upon",
    "into",
    "unto",
    "hath",
    "doth",
    "thou",
    "thee",
    "thy",
}


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().strip('"')).strip()


def is_usable(line: str) -> bool:
    if len(line) < 12:
        return False
    if not re.search(r"[a-z]", line, re.I):
        return False
    return not any(pattern.search(line) for pattern in STAGE_PATTERNS)


def tokenize(line: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]", line)]


def word_tokens(line: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z]+(?:'[a-z]+)?", line.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def read_lines_from_zip() -> list[str]:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())

        if "alllines.txt" in names:
            raw = archive.read("alllines.txt").decode("utf-8", errors="ignore")
            return [clean_line(line) for line in raw.splitlines()]

        if "Shakespeare_data.csv" in names:
            raw = archive.read("Shakespeare_data.csv").decode("utf-8", errors="ignore").splitlines()
            reader = csv.DictReader(raw)
            return [clean_line(row.get("PlayerLine", "")) for row in reader]

    raise FileNotFoundError("The zip must contain alllines.txt or Shakespeare_data.csv.")


def add_weighted(target: dict[str, Counter], key: str, value: str) -> None:
    target[key][value] += 1


def compact_counter(counter: Counter, limit: int) -> list[list[str | int]]:
    return [[token, count] for token, count in counter.most_common(limit)]


def build_ngram(lines: list[str]) -> tuple[dict[str, Counter], dict[str, Counter], Counter]:
    transitions: dict[str, Counter] = defaultdict(Counter)
    starters: dict[str, Counter] = defaultdict(Counter)
    vocabulary: Counter = Counter()

    for line in lines:
        tokens = tokenize(line)
        if len(tokens) < 5:
            continue

        vocabulary.update(token for token in tokens if re.match(r"^[a-z]", token))
        padded = ["<s>", "<s>", *tokens, "</s>"]
        add_weighted(starters, "<s> <s>", padded[2])

        for index in range(len(padded) - ORDER):
            key = " ".join(padded[index : index + ORDER])
            add_weighted(transitions, key, padded[index + ORDER])

    return transitions, starters, vocabulary


def compact_weighted_map(weighted_map: dict[str, Counter]) -> dict[str, list[list[str | int]]]:
    return {
        key: compact_counter(counter, MAX_CHOICES_PER_STATE)
        for key, counter in weighted_map.items()
    }


def evaluate_ngram(transitions: dict[str, Counter], eval_lines: list[str], vocabulary_size: int) -> dict:
    total = 0
    covered = 0
    top_1 = 0
    top_5 = 0
    negative_log_likelihood = 0.0
    smoothing = 0.1

    for line in eval_lines:
        tokens = tokenize(line)
        if len(tokens) < 5:
            continue

        padded = ["<s>", "<s>", *tokens, "</s>"]

        for index in range(len(padded) - ORDER):
            key = " ".join(padded[index : index + ORDER])
            target = padded[index + ORDER]
            choices = transitions.get(key)
            total += 1

            if not choices:
                negative_log_likelihood -= math.log(1 / max(vocabulary_size, 1))
                continue

            covered += 1
            ranked = [token for token, _ in choices.most_common(5)]
            if ranked and ranked[0] == target:
                top_1 += 1
            if target in ranked:
                top_5 += 1

            denominator = sum(choices.values()) + smoothing * max(vocabulary_size, 1)
            probability = (choices.get(target, 0) + smoothing) / denominator
            negative_log_likelihood -= math.log(probability)

    perplexity = math.exp(negative_log_likelihood / max(total, 1))

    return {
        "type": "held-out next-token evaluation",
        "evalLines": len(eval_lines),
        "tokens": total,
        "coverage": round(covered / max(total, 1), 4),
        "top1Accuracy": round(top_1 / max(total, 1), 4),
        "top5Accuracy": round(top_5 / max(total, 1), 4),
        "perplexity": round(perplexity, 2),
    }


def build_seed_index(transitions: dict, vocabulary: Counter) -> dict[str, list[str]]:
    common = {word for word, _ in vocabulary.most_common(1200)}
    seed_index: dict[str, list[str]] = defaultdict(list)

    for state in transitions:
        for token in state.split():
            if token in common and len(token) > 3 and len(seed_index[token]) < MAX_STATES_PER_SEED:
                seed_index[token].append(state)

    return dict(seed_index)


def build_retrieval(lines: list[str]) -> dict:
    chunks = [
        " ".join(lines[index : index + LINES_PER_DOC])
        for index in range(0, len(lines), LINES_PER_DOC)
    ]
    chunks = [chunk for chunk in chunks if len(chunk) > 80]

    document_tokens = [word_tokens(chunk) for chunk in chunks]
    document_frequency: Counter = Counter()
    for tokens in document_tokens:
        document_frequency.update(set(tokens))

    total_docs = len(document_tokens)
    idf = {
        term: math.log((1 + total_docs) / (1 + count)) + 1
        for term, count in document_frequency.items()
        if count >= 3
    }

    docs = []
    for index, (chunk, tokens) in enumerate(zip(chunks, document_tokens)):
        counts = Counter(token for token in tokens if token in idf)
        if not counts:
            continue

        terms = sorted(
            ((term, round((count / len(tokens)) * idf[term], 6)) for term, count in counts.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:16]

        docs.append(
            {
                "id": index,
                "text": chunk[:520],
                "terms": terms,
            }
        )

    docs.sort(key=lambda doc: sum(score for _, score in doc["terms"]), reverse=True)
    return {
        "method": "tf-idf chunk retrieval",
        "sourceDocs": total_docs,
        "docs": docs[:MAX_RETRIEVAL_DOCS],
    }


def main() -> None:
    lines = [line for line in read_lines_from_zip() if is_usable(line)]
    eval_lines = lines[::10]
    train_lines = [line for index, line in enumerate(lines) if index % 10 != 0]
    eval_transitions, _, eval_vocabulary = build_ngram(train_lines)
    metrics = evaluate_ngram(eval_transitions, eval_lines, len(eval_vocabulary))

    transitions, starters, vocabulary = build_ngram(lines)
    retrieval = build_retrieval(lines)
    seed_index = build_seed_index(transitions, vocabulary)

    model = {
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "source": str(ZIP_PATH),
        "lineCount": len(lines),
        "concepts": [
            "NLP cleaning and tokenization",
            "word-level n-gram language model",
            "TF-IDF retrieval index for RAG-style context selection",
            "intent-conditioned response framing",
        ],
        "metrics": metrics,
        "generator": {
            "order": ORDER,
            "transitions": compact_weighted_map(transitions),
            "starters": compact_weighted_map(starters),
            "seedIndex": seed_index,
            "commonWords": [word for word, _ in vocabulary.most_common(900)],
        },
        "retrieval": retrieval,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    print(f"Trained Python Shakespeare model from {len(lines)} lines.")
    print(
        "Held-out next-token accuracy: "
        f"top-1={metrics['top1Accuracy']:.2%}, "
        f"top-5={metrics['top5Accuracy']:.2%}, "
        f"coverage={metrics['coverage']:.2%}, "
        f"perplexity={metrics['perplexity']}"
    )
    print(f"Built {len(retrieval['docs'])} retrieval chunks from {retrieval['sourceDocs']} source chunks.")
    print(f"Wrote {MODEL_PATH}.")


if __name__ == "__main__":
    main()
