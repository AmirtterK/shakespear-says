// Regression test for the bugs we found and fixed:
//  1. Output truncated to 1-2 words (prompt-slicing bug)
//  2. Output chopped down to a lone interjection like "Hark!" (premature sentence-cut)
//  3. Words glued together like "hisShe" (quote-strip bug)
//  4. Leftover literal `"` characters in the final text
//
// Usage: node test_generation.mjs   (requires the dev server running on :3000)

const BASE_URL = "http://localhost:3000";

const PROMPTS = [
  "mom",
  "dad",
  "insult a lazy programmer",
  "advice for a new job",
  "a beautiful sunset",
  "insult my cat",
  "compliment a friend",
  "warn a soldier",
  "love",
  "war",
];

const MIN_WORDS = 3; // catches the 1-2 word truncation regression
const MAX_CHARS = 260; // catches unbounded run-ons

function checkGluedWords(text) {
  // Heuristic: a lowercase letter immediately followed by an uppercase letter with
  // no space between usually means two words got welded together (e.g. "hisShe").
  return /[a-z][A-Z]/.test(text);
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
  if (checkGluedWords(text)) failures.push(`possible glued words (lowercase immediately followed by uppercase)`);
  if (text === "Verily, thy words hath moved my soul.") failures.push(`hit the empty-generation fallback string`);

  return { prompt, text, words: words.length, chars: text.length, failures };
}

async function main() {
  console.log(`Running ${PROMPTS.length} prompts against ${BASE_URL} ...\n`);
  const results = [];
  for (const p of PROMPTS) {
    const r = await runOne(p);
    results.push(r);
    const status = r.failures.length === 0 ? "PASS" : "FAIL";
    console.log(`[${status}] "${p}" -> "${r.text}"  (${r.words} words, ${r.chars} chars)`);
    if (r.failures.length) {
      for (const f of r.failures) console.log(`         - ${f}`);
    }
  }

  const passed = results.filter((r) => r.failures.length === 0).length;
  console.log(`\n${passed}/${results.length} passed.`);
  if (passed !== results.length) process.exitCode = 1;
}

main().catch((err) => {
  console.error("Test run crashed:", err);
  process.exitCode = 1;
});
