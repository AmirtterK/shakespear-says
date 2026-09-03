
Shakespeare Says
=================

![Shakespeare](public/shakespeare.jpg)

Shakespeare Says is a small web app with a single button: press it, and a
GPT-2 model fine-tuned on Shakespeare's plays generates a random line of
Shakespearean-style text.

## Idea

The goal was to see how far a small language model, trained only on the
public-domain text of Shakespeare's plays, could go in mimicking his voice
and manner of speaking -- not quoting or copy-pasting existing lines, but
generating new ones in a similar style: archaic vocabulary, inverted
sentence structure, and the rhythm of Elizabethan dialogue.

## How it works

- **Model**: a GPT-2 model fine-tuned from scratch on the full text of
  Shakespeare's plays (`scripts/train_gpt2.py`). The fine-tuned weights are
  exported to ONNX (`shakespeare-gpt2-onnx`) so they can run directly in the
  Next.js server without a Python backend.
- **Generation**: the app seeds the model with a short few-shot prompt built
  from a random word, then samples a continuation from it
  (`src/app/api/generate/route.ts`). The raw output is cleaned up afterward
  to strip leftover formatting artifacts the model picked up from the
  training data (stage directions, act/scene headers, character names) and
  to trim it down to a single line.
- **Frontend**: a single page (`src/app/page.tsx`) with one button. Each
  press calls the generation API and displays the resulting line.
- There is also an earlier, simpler word-level n-gram model
  (`scripts/train_shakespeare.py`) kept in the repo for reference; the app
  itself runs on the fine-tuned GPT-2 model.

## Running locally

The fine-tuned model weights are not stored in this repository (they are
large binary files, kept locally instead -- see `.gitignore`). To run the
app you need the model files present under `src/lib/shakespeare-gpt2-onnx`
on your own machine.

```bash
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Training

If you want to retrain the model from the Shakespeare dataset:

```bash
npm run train:gpt2
```

This reads `shakespear.zip`, extracts and cleans the dialogue lines, and
fine-tunes a base GPT-2 model on them, saving the result to
`src/lib/shakespeare-gpt2`.
