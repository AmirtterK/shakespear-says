import { pipeline, env } from "@xenova/transformers";
import path from "path";

// Force transformers.js to load from disk, not the Hugging Face Hub
env.allowRemoteModels = false;
env.localModelPath = path.join(process.cwd(), "src", "lib");

let generator: any = null;

async function initModel() {
  if (!generator) {
    console.log("[initModel] start, generator not yet cached");
    console.log("Loading local Shakespeare GPT-2 model...");
    const t0 = Date.now();
    // Load from local folder: src/lib/shakespeare-gpt2-onnx
    generator = await pipeline(
      "text-generation",
      "shakespeare-gpt2-onnx",
      {
        // Confirmed fp32 (quantized: false) and the quantized model produce
        // identical output quality (same underlying fine-tuned weights) --
        // fp32 is just ~5x slower and once OOM-crashed the dev server. So
        // quantized is strictly better here.
        quantized: true,
        progress_callback: (progress: any) => {
          console.log("[initModel] progress:", JSON.stringify(progress));
        },
      }
    );
    console.log(`[initModel] pipeline() resolved in ${Date.now() - t0}ms`);
  } else {
    console.log("[initModel] using cached generator");
  }
  return generator;
}

const FEW_SHOT_EXAMPLES = [
  { word: "apple", line: "What fruit is this, so crimson and forbidden, that tempted Eve herself to sin?" },
  { word: "dog", line: "Beware the dog that fawns and barks by day, for silent teeth may bite thee in the night." },
  { word: "war", line: "When drums of war do beat, all peace lies slain upon the trampled field." },
  { word: "king", line: "Uneasy lies the head that wears the crown, for kings must reign in constant dread." },
  { word: "storm", line: "The storm doth rage as fierce as any heart in torment torn." },
  { word: "mother", line: "A mother's love doth burn as steady as the northern star, unmoved by time or tempest." },
  { word: "gold", line: "O cursed gold, for thee do men betray their kin and sell their very souls." },
  { word: "moon", line: "The moon doth hang, a silver coin upon the velvet purse of night." },
  { word: "fool", line: "A fool doth think himself most wise, while wisdom knows itself a fool." },
  { word: "sword", line: "This sword hath drunk the blood of kings, and thirsts for one draught more." },
  { word: "garden", line: "In this fair garden every rose doth hide a thorn beneath its sweetness." },
  { word: "wine", line: "Wine doth loosen every tongue and drown a thousand secrets in its cup." },
];

function buildFewShotPrompt(word: string) {
  const examples = FEW_SHOT_EXAMPLES.map(
    (e) => `Word: ${e.word}\nShakespeare says: "${e.line}"`
  ).join("\n\n");
  return `${examples}\n\nWord: ${word}\nShakespeare says: "`;
}

// No user input anymore -- the button just asks for a random line. We still
// drive it off the proven few-shot/word scaffold (that's what produces clean,
// well-formed Shakespearean sentences), we just pick the word ourselves
// instead of taking it from a prompt.
const RANDOM_SEED_WORDS = [
  ...FEW_SHOT_EXAMPLES.map((e) => e.word),
  "river", "silence", "ghost", "thunder", "betrayal", "ocean", "fire",
  "mountain", "shadow", "dance", "hunger", "friendship", "storm", "mirror",
  "dream", "poison", "crown", "whisper", "table",
];

function pickRandomSeedWord() {
  return RANDOM_SEED_WORDS[Math.floor(Math.random() * RANDOM_SEED_WORDS.length)];
}

export async function POST() {
  console.log("[POST /api/generate] request received");
  try {
    const seedWord = pickRandomSeedWord();
    console.log("[POST /api/generate] seed word:", seedWord);

    console.log("[POST /api/generate] calling initModel()...");
    const gen = await initModel();
    console.log("[POST /api/generate] initModel() returned, generator ready");

    const fewShotPrompt = buildFewShotPrompt(seedWord);

    console.log("[POST /api/generate] calling generator...");
    const t1 = Date.now();
    const results = await gen(fewShotPrompt, {
      max_new_tokens: 27, // was 40 -- shorter budget makes it less likely to run past one clause into a second, unrelated one
      min_new_tokens: 12, // don't let EOS/quote end generation after just 1-2 tokens
      do_sample: true,
      temperature: 0.7, // was 0.8 -- less drift
      top_p: 0.85, // was 0.9 -- less drift
      repetition_penalty: 1.3,
      no_repeat_ngram_size: 3,
      // Let the tokenizer strip the prompt via token offsets instead of us slicing
      // generatedText by fewShotPrompt.length (raw JS string length). GPT-2's decode
      // step (clean_up_tokenization_spaces) doesn't always reproduce the original
      // prompt string byte-for-byte -- spacing around quotes/colons can shift -- so
      // that manual slice landed in the wrong spot and usually grabbed the tail end
      // of the prompt itself (which has a stray quote right there), which is why
      // output was getting chopped down to 1-2 words.
      return_full_text: false,
    });
    console.log(`[POST /api/generate] generation finished in ${Date.now() - t1}ms`);

    const continuation: string = results[0]?.generated_text || "";

    // This fine-tuned model sprinkles stray `"` and `:` throughout its output -- an
    // artifact of the play-script dialogue (character names, stage directions) it was
    // trained on -- so cutting at the first `"` (the old approach) chopped the reply
    // down to 1-2 words almost every time, even though the model had generated a full
    // 40 tokens. Instead: find the boundary of the *next few-shot example* (a newline,
    // or the literal "Word:" marker), strip stray quote characters from what's left,
    // then trim down to the first full sentence.
    const boundaryPoints = [
      continuation.indexOf("\n"),
      continuation.indexOf("Word:"),
    ].filter((i) => i !== -1);
    const boundaryAt = boundaryPoints.length > 0 ? Math.min(...boundaryPoints) : continuation.length;

    let cleaned = continuation
      .slice(0, Math.max(0, boundaryAt))
      .replace(/"/g, " ") // space, not "" -- a quote sitting between two words with no
      // surrounding whitespace (e.g. `his"She`) would otherwise weld them into
      // `hisShe`; the following \s+ -> " " collapses any doubled-up spacing this adds
      .replace(/\s+/g, " ")
      .trim();

    // The training data was raw play-script lines (stage directions, act/scene
    // headers, speaker names all mixed in with actual dialogue, no separation) --
    // so the model itself learned to reproduce that formatting noise as part of
    // its "style". We can't un-teach that without retraining, so strip it out of
    // the output here instead: remove bracket/paren stage notes, ACT/SCENE
    // headers, stage-direction verbs ("Enter", "Exeunt", ...) and runs of
    // ALL-CAPS words (character names / speaker tags like "CLIFFORD" or
    // "ALAN HUGH WESTORSDALE").
    cleaned = cleaned
      .replace(/\[[^\]]*\]?/g, " ")
      .replace(/\([^)]*\)?/g, " ")
      .replace(/[\[\]()]/g, " ") // safety net: strip any stray unmatched bracket the paired removals above missed
      .replace(/\b(?:ACT|SCENE)\s+[IVXLCDM]+\b\.?/gi, " ")
      .replace(/\b(?:Enter|Exit|Exeunt|Re-enter|Alarum|Flourish|Prologue)\b[^.!?:;,\-]*/gi, " ")
      .replace(/\b(?:[A-Z]{2,}\s+){1,3}[A-Z]{2,}\b/g, " ") // multi-word ALL-CAPS name runs
      .replace(/\b[A-Z]{2,}\b/g, " ") // single ALL-CAPS words (e.g. a lone speaker tag)
      .replace(/\s+([,.;:!?])/g, "$1") // no space before punctuation left behind by a removal
      .replace(/[,;:\-]+(?=[.!?])/g, "") // drop stray punctuation sitting right before a terminal mark, e.g. "song:."
      .replace(/([,;:\-])(?:\s*[,;:\-])+/g, "$1") // collapse doubled-up punctuation from adjacent removals
      .replace(/\s+/g, " ")
      .replace(/^[\s,;:\-]+/, "")
      .replace(/[\s,;:\-]+$/, "")
      .trim();

    // NOTE: we used to cut down to the first full sentence here (first ".", "!", or
    // "?"). That was a regression the first time around -- this model loves short
    // Shakespearean interjections ("Hark!", "Fie!", "Pray!") that show up in the
    // first few words and already end in "!", so a naive "stop at first sentence-end"
    // rule kept chopping every reply down to just the interjection.
    //
    // This version avoids that: it only treats [.!?] as a real sentence boundary if
    // (a) it comes after a minimum length (so early interjections don't qualify) and
    // (b) it's followed by a capital letter or the end of the string (so it's not,
    // say, a "Mr." or an abbreviation mid-clause). If no such boundary is found, we
    // fall back to the existing length-based trim below -- so this can only ever
    // shorten output further, never reintroduce the old bug on a fresh generation.
    const MIN_SENTENCE_LEN = 25;
    const sentenceEndRe = /[.!?](?=\s+[A-Z]|\s*$)/g;
    let sentenceCut = -1;
    let match: RegExpExecArray | null;
    while ((match = sentenceEndRe.exec(cleaned)) !== null) {
      if (match.index >= MIN_SENTENCE_LEN) {
        sentenceCut = match.index + 1; // include the punctuation itself
        break;
      }
    }
    if (sentenceCut !== -1) {
      cleaned = cleaned.slice(0, sentenceCut).trim();
    } else if (cleaned.length > 220) {
      const lastBreak = cleaned.slice(0, 220).lastIndexOf(" ");
      cleaned = cleaned.slice(0, lastBreak > 0 ? lastBreak : 220).trim();
    }

    const text = cleaned;

    return Response.json({
      text: text || "Verily, thy words hath moved my soul.",
      model: "GPT-2 Fine-tuned on Shakespeare",
    });
  } catch (error) {
    // Log as a plain string only — passing the raw error object to
    // console.error/console.log can trip Next's terminal code-frame
    // formatter and throw a second, unrelated error that swallows
    // this response before it's sent.
    const message = error instanceof Error ? error.message : String(error);
    const stack = error instanceof Error ? error.stack : undefined;
    process.stdout.write(`[POST /api/generate] Generation error: ${message}\n`);
    if (stack) process.stdout.write(stack + "\n");

    return Response.json(
      {
        error: "Generation failed.",
        details: message,
      },
      { status: 500 }
    );
  }
}
