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

function detectIntent(prompt: string) {
  const value = prompt.toLowerCase();
  if (/\b(insult|roast|mock|curse|fool|lazy|stupid)\b/.test(value)) return "insult";
  if (/\b(compliment|praise|love|romance|beautiful|handsome)\b/.test(value)) return "compliment";
  if (/\b(advice|wisdom|teach|guide)\b/.test(value)) return "advice";
  if (/\b(threat|warn|fight|battle|challenge)\b/.test(value)) return "threat";
  return "default";
}

export async function POST(request: Request) {
  console.log("[POST /api/generate] request received");
  try {
    const body = (await request.json().catch(() => null)) as {
      prompt?: string;
    } | null;
    const prompt = body?.prompt?.trim();
    console.log("[POST /api/generate] prompt:", prompt);

    if (!prompt) {
      console.log("[POST /api/generate] missing prompt, returning 400");
      return Response.json({ error: "Prompt is required." }, { status: 400 });
    }

    console.log("[POST /api/generate] calling initModel()...");
    const gen = await initModel();
    console.log("[POST /api/generate] initModel() returned, generator ready");

    const fewShotPrompt = buildFewShotPrompt(prompt);

    console.log("[POST /api/generate] calling generator...");
    const t1 = Date.now();
    const results = await gen(fewShotPrompt, {
      max_new_tokens: 40,
      min_new_tokens: 12, // don't let EOS/quote end generation after just 1-2 tokens
      do_sample: true,
      temperature: 0.8,
      top_p: 0.9,
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
      .replace(/"/g, "")
      .replace(/\s+/g, " ")
      .trim();

    // Prefer stopping at the end of the first full sentence, when there is one.
    const sentenceMatch = cleaned.match(/^.*?[.!?]/);
    if (sentenceMatch && sentenceMatch[0].trim().length > 3) {
      cleaned = sentenceMatch[0].trim();
    }

    const text = cleaned;

    return Response.json({
      intent: detectIntent(prompt),
      subject: prompt.split(" ").slice(0, 3).join(" "),
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
