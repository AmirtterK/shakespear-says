
Shakespeare Says
=================

![Shakespeare](public/shakespeare.jpg)

Shakespeare Says is a small web app with a single button: press it, and a
GPT-2 model fine-tuned on Shakespeare's plays generates a short, new passage
in his style.

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

## Running locally

The model weights are not in git (see `.gitignore`). Put the exported model in
`src/lib/shakespeare-gpt2-onnx/` (see "Retraining" below), then:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). With the dev server
running, `node test_generation.mjs` generates 20 lines and checks them for
formatting problems, repeats, and leaked stage directions. Always read the
samples it prints too.

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
