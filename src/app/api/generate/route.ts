import { getModel } from "@/lib/model-manager";

// The model was fine-tuned on samples shaped like "SPEAKER:\nline\nline<eos>"
// (see scripts/prepare_data.py). Prompting with just a speaker header is exactly
// what it saw in training. These names all exist in the training data.
const SPEAKERS = [
  "HAMLET", "PORTIA", "KING LEAR", "FALSTAFF", "JULIET", "ROMEO", "IAGO", "OTHELLO",
  "MACBETH", "LADY MACBETH", "PROSPERO", "ROSALIND", "CLEOPATRA", "MARK ANTONY",
  "BRUTUS", "CASSIUS", "SHYLOCK", "BENEDICK", "VIOLA", "KING HENRY V",
  "KING RICHARD III", "PETRUCHIO", "DESDEMONA", "ISABELLA", "TIMON", "CORIOLANUS",
  "TITUS ANDRONICUS", "KING CLAUDIUS", "LEONTES", "HOTSPUR", "EDGAR", "KENT",
  "FRIAR LAURENCE", "LORD POLONIUS", "PERICLES", "DUKE VINCENTIO", "HELENA", "IMOGEN",
];

const MAX_LINES = 3;
const MAX_ATTEMPTS = 3;

function pickSpeaker() {
  return SPEAKERS[Math.floor(Math.random() * SPEAKERS.length)];
}

function tidy(raw: string): string {
  let lines = raw
    .split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);

  // If the model started a new "SPEAKER:" block, keep only the first speech.
  const nextSpeech = lines.findIndex((line, index) => index > 0 && /^[A-Z][A-Z .'-]{2,}:$/.test(line));
  if (nextSpeech !== -1) lines = lines.slice(0, nextSpeech);

  lines = lines.slice(0, MAX_LINES);

  // Generation stops at <eos>, but if it hit the token limit the last line may be cut off.
  const endsCleanly = (line: string) => /[.!?;:]["']?$/.test(line);
  if (lines.length > 1 && !endsCleanly(lines[lines.length - 1])) {
    lines = lines.slice(0, -1);
  }

  let text = lines.join("\n");
  text = text.replace(/[,;:\s-]+$/, ".");
  if (!/[.!?]["']?$/.test(text)) text += ".";
  return text;
}

function isUsable(text: string): boolean {
  if (text.length < 25) return false;
  if (text.split(/\s+/).length < 5) return false;
  if (/[\[\]{}<>]/.test(text)) return false;
  if (/\b[A-Z]{3,}\b/.test(text)) return false; // a speaker name or stage direction leaked in
  return true;
}

export async function POST() {
  try {
    const generator = await getModel();

    let best = "";
    let speaker = pickSpeaker();

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      speaker = pickSpeaker();
      const prompt = `${speaker}:\n`;

      const results = await generator(prompt, {
        max_new_tokens: 60,
        do_sample: true,
        temperature: 0.85,
        top_k: 50,
        top_p: 0.92,
        repetition_penalty: 1.1,
        return_full_text: false,
      });

      let generated: string = results?.[0]?.generated_text ?? "";
      if (generated.startsWith(prompt)) generated = generated.slice(prompt.length);

      const text = tidy(generated);
      if (!best) best = text; // fallback if every attempt fails the quality gate
      if (isUsable(text)) {
        return Response.json({ text, speaker, model: "GPT-2 fine-tuned on Shakespeare" });
      }
    }

    if (best) {
      return Response.json({ text: best, speaker, model: "GPT-2 fine-tuned on Shakespeare" });
    }
    return Response.json({ error: "The model produced no usable text. Try again." }, { status: 502 });
  } catch (error) {
    // Log as a plain string: passing the raw error object to console.error can trip
    // Next's terminal code-frame formatter and swallow the response.
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`[generate] error: ${message}\n`);
    return Response.json({ error: "Generation failed.", details: message }, { status: 500 });
  }
}
