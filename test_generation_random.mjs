// Extended regression test: many random single/few-word prompts, checking
// structural bugs, polish signals, AND a rough relevance heuristic (does the
// output ever engage with the actual prompt word, or a close variant of it?).
//
// Usage: node test_generation_random.mjs   (requires the dev server running on :3000)

const BASE_URL = "http://localhost:3000";

const PROMPTS = [
  "table", "river", "silence", "ghost", "money", "thunder", "garden",
  "betrayal", "king", "ocean", "fire", "forgive", "mountain", "shadow",
  "dance", "hunger", "moon", "sword", "friendship", "storm", "mirror",
  "dream", "poison", "crown", "whisper",
];

const MIN_WORDS = 3;
const MAX_CHARS = 260;

function checkGluedWords(text) {
  return /[a-z][A-Z]/.test(text);
}
function checkDoubleSpace(text) {
  return /  /.test(text);
}
function checkOrphanPunctuation(text) {
  return /\s['"\-,;:]\s/.test(text);
}
function checkIsolatedSingleLetter(text) {
  return /(^|\s)[b-hj-zB-HJ-Z](\s|$)/.test(text);
}
function checkLeadingTrailingPunct(text) {
  return /^[\s,;:\-]|[,;:\-]$/.test(text.trim());
}
function checkRepeatedPunct(text) {
  return /([!?.,;:])\1/.test(text) || /[!?]{2,}/.test(text);
}

// Rough relevance heuristic: does the output contain the prompt word itself,
// or share a >=4-letter stem with it (catches "storm"/"storms"/"stormy")?
// This is a weak proxy, not real semantic relevance -- it only catches the
// most literal case of topic-tracking, but a fine-tuned GPT-2 that's actually
// anchoring to the word should clear this most of the time.
function checkRelevance(prompt, text) {
  const p = prompt.toLowerCase();
  const t = text.toLowerCase();
  if (t.includes(p)) return true;
  if (p.length >= 4) {
    const stem = p.slice(0, Math.min(p.length, 5));
    const words = t.split(/\W+/);
    return words.some((w) => w.length >= 4 && w.startsWith(stem));
  }
  return false;
}

async function runOne(prompt) {
  const res = await fetch(`${BASE_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const data = await res.json();
  const text = data.text || "";
  const words = text.trim().split(/\s+/).filter(Boolean);

  const failures = [];
  if (!res.ok) failures.push(`HTTP ${res.status}: ${JSON.stringify(data)}`);
  if (words.length < MIN_WORDS) failures.push(`too short (${words.length} words)`);
  if (text.length > MAX_CHARS) failures.push(`too long (${text.length} chars)`);
  if (text.includes('"')) failures.push(`leftover literal quote character`);
  if (checkGluedWords(text)) failures.push(`possible glued words`);
  if (text === "Verily, thy words hath moved my soul.") failures.push(`hit the empty-generation fallback string`);

  const polishNotes = [];
  if (checkDoubleSpace(text)) polishNotes.push(`double space`);
  if (checkOrphanPunctuation(text)) polishNotes.push(`orphaned punctuation mark`);
  if (checkIsolatedSingleLetter(text)) polishNotes.push(`isolated single-letter token`);
  if (checkLeadingTrailingPunct(text)) polishNotes.push(`leading/trailing punctuation`);
  if (checkRepeatedPunct(text)) polishNotes.push(`repeated punctuation`);

  const relevant = checkRelevance(prompt, text);

  return { prompt, text, words: words.length, chars: text.length, failures, polishNotes, relevant };
}

async function main() {
  console.log(`Running ${PROMPTS.length} prompts against ${BASE_URL} ...\n`);
  const results = [];
  for (const p of PROMPTS) {
    const r = await runOne(p);
    results.push(r);
    const status = r.failures.length === 0 ? "PASS" : "FAIL";
    console.log(`[${status}] "${p}" -> "${r.text}"`);
    console.log(`         (${r.words} words, ${r.chars} chars) relevance: ${r.relevant ? "word present" : "no trace of prompt word"}`);
    if (r.failures.length) {
      for (const f of r.failures) console.log(`         FAIL: ${f}`);
    }
    if (r.polishNotes.length) {
      console.log(`         polish flags: ${r.polishNotes.join(", ")}`);
    }
  }

  const passed = results.filter((r) => r.failures.length === 0).length;
  const clean = results.filter((r) => r.failures.length === 0 && r.polishNotes.length === 0).length;
  const relevant = results.filter((r) => r.relevant).length;
  console.log(`\n${passed}/${results.length} passed structural checks.`);
  console.log(`${clean}/${results.length} had zero polish flags.`);
  console.log(`${relevant}/${results.length} contained any trace of the prompt word (weak relevance proxy).`);
  if (passed !== results.length) process.exitCode = 1;
}

main().catch((err) => {
  console.error("Test run crashed:", err);
  process.exitCode = 1;
});
