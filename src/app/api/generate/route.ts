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
        quantized: false, // we exported a plain fp32 model.onnx, not model_quantized.onnx
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
      do_sample: true,
      temperature: 0.8,
      top_p: 0.9,
    });
    console.log(`[POST /api/generate] generation finished in ${Date.now() - t1}ms`);

    const generatedText = results[0].generated_text;
    // The pipeline returns fewShotPrompt + continuation, so slice by
    // index rather than .replace (safer since the prompt text can
    // legitimately reappear inside the continuation).
    let continuation = generatedText.slice(fewShotPrompt.length);
    // Cut at the closing quote, a newline, or the start of the next
    // "Word:" example — whichever the model reaches first.
    const cutPoints = [
      continuation.indexOf('"'),
      continuation.indexOf("\n"),
      continuation.indexOf("Word:"),
    ].filter((i) => i !== -1);
    const cutAt = cutPoints.length > 0 ? Math.min(...cutPoints) : continuation.length;
    const text = continuation.slice(0, cutAt).trim();

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
