# Shakespeare Says

<img src="public/shakespeare.png" alt="Shakespeare" width="120">

A text generator that produces Shakespearean-style lines, powered by a GPT-2 model fine-tuned on Shakespeare's works and running locally through a quantized ONNX model.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AmirtterK/shakespear-says.git
cd shakespear-says
```

### 2. Download the ONNX model

The model weights are **not** included in the repo (they're ~204 MB).
Download the zip from GitHub Releases:

> **[Download shakespeare-gpt2-onnx.zip](https://github.com/AmirtterK/shakespear-says/releases/download/v1/shakespeare-gpt2-onnx.zip)**

### 3. Extract the model

Unzip the downloaded file into `src/lib/` so the folder structure looks like
this:

```
src/
└── lib/
    └── shakespeare-gpt2-onnx/
        ├── config.json
        ├── generation_config.json
        ├── merges.txt
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.json
        └── onnx/
            └── model_quantized.onnx
```

> **Tip:** If the zip extracts into a nested folder, make sure
> `shakespeare-gpt2-onnx/` sits directly inside `src/lib/`.

### 4. Install dependencies

```bash
npm install
```

### 5. Run the development server

```bash
npm run dev
```

### 6. Open the app

Navigate to **[http://localhost:3000](http://localhost:3000)** in your browser.

The model loads automatically on first visit — you'll see a loading orb and
progress bar. Once loaded, hit **Generate** to summon an Elizabethan passage.

---

## How it works

- **Data** (`scripts/prepare_data.py`): `data/raw/Shakespeare_data.csv` is
  cleaned into speaker-formatted samples. Stage directions, act/scene headers
  and inline `[Aside]`-style notes are dropped; lines are grouped into speeches
  and cut into chunks of up to 4 lines:

  ```
  HAMLET:
  To be, or not to be, that is the question:
  Whether 'tis nobler in the mind to suffer
  ```

- **Model** (`scripts/train_gpt2.py`): GPT-2 (124M) fine-tuned from the base
  `gpt2` weights on those samples, each ending in a real `<eos>` so the model
  learns to stop. A 5% held-out validation set picks the best epoch.
- **Export** (`scripts/export_onnx.py`): the checkpoint is exported to ONNX
  with `optimum`, int8-quantized, and checked against the fp32 export
  (perplexity + greedy continuations). The output folder matches what
  `@xenova/transformers` loads: `src/lib/shakespeare-gpt2-onnx/`.
- **Generation** (`src/app/api/generate/route.ts`): the API prompts the model
  with just a random speaker header (`"PORTIA:\n"`), the same format it was
  trained on, samples with moderate settings (temperature 0.85, top-k 50,
  top-p 0.92, repetition penalty 1.1), and lightly tidies the result.
- **Frontend**: a single page (`src/app/page.tsx`) with one button.
- `scripts/train_shakespeare.py` is an older word-level n-gram model kept for
  reference; the app does not use it.

---

## Testing

With the dev server running:

```bash
node test_generation.mjs
```

This generates 20 lines and checks them for formatting problems, repeats, and
leaked stage directions.

---

## Retraining (Google Colab, free T4 GPU)

1. Open `colab/shakespeare_says_retrain.ipynb` in Colab and switch the
   runtime to a T4 GPU.
2. Run the cells. When asked to upload files, pick `data/raw/Shakespeare_data.csv`
   and the three scripts `prepare_data.py`, `train_gpt2.py`, `export_onnx.py`.
3. Download `shakespeare-gpt2-onnx.zip` at the end, unzip it into `src/lib/`
   (replacing the old `shakespeare-gpt2-onnx` folder), and restart `npm run dev`.

The same scripts also run locally (`npm run prepare:data`, `npm run train:gpt2`,
`npm run export:onnx`) if you have a CUDA GPU and the pinned packages from the
notebook: `transformers==4.46.3`, `optimum[onnxruntime]==1.23.3`,
`onnx==1.17.0`, `onnxruntime==1.20.1`, `torch==2.5.1`.

Notes on the export: by default only matrix multiplications are quantized
(`QUANT_OPS=MatMul`), which keeps quality close to fp32 at the cost of a larger
file than quantizing embeddings too. Set `INCLUDE_FP32=1` to also ship the fp32
model, or `QUANT_OPS=MatMul,Gather` for a smaller, lower-quality file.
