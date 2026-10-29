// Quality check for /api/generate.
//
// The old version of this test only checked length and stray quotes, so it reported
// 10/10 PASS on gibberish. This one checks that the output actually looks like clean
// verse and that the model isn't repeating itself, then prints every sample so you
// can read them yourself -- that's still the most important check.
//
// Usage: node test_generation.mjs [count]   (requires the dev server running on :3000)

const BASE_URL = process.env.BASE_URL ?? "http://localhost:3000";
const COUNT = Number(process.argv[2] ?? 20);

const MIN_WORDS = 5;
const MAX_WORDS = 60;
const MAX_PUNCT_RATIO = 0.12; // share of characters that are not letters/digits/space/apostrophe
const MAX_FAIL_RATE = 0.1;
const MIN_UNIQUE_RATE = 0.9;

function checks(text) {
  const failures = [];
  const words = text.trim().split(/\s+/).filter(Boolean);
  const punct = (text.match(/[^A-Za-z0-9\s']/g) ?? []).length;

  if (words.length < MIN_WORDS) failures.push(`too short (${words.length} words)`);
  if (words.length > MAX_WORDS) failures.push(`too long (${words.length} words)`);
  if (/["\[\]{}()<>]/.test(text)) failures.push("contains quote/bracket characters");
  if (/--/.test(text)) failures.push("contains '--'");
  if (/\b[A-Z]{3,}\b/.test(text)) failures.push("contains an ALL-CAPS word (speaker name / stage direction)");
  if (/[a-z][A-Z]/.test(text)) failures.push("glued words (lowercase followed by uppercase)");
  if (!/[.!?]$/.test(text.trim())) failures.push("does not end with . ! or ?");
  if (/[,;:]\s*[,;:.!?]/.test(text)) failures.push("doubled-up punctuation");
  if (text.length > 0 && punct / text.length > MAX_PUNCT_RATIO) failures.push("too much punctuation");
  return failures;
}

async function runOne() {
  const res = await fetch(`${BASE_URL}/api/generate`, { method: "POST" });
  const data = await res.json();
  const text = data.text ?? "";
  const failures = checks(text);
  if (!res.ok) failures.unshift(`HTTP ${res.status}: ${JSON.stringify(data)}`);
  return { text, speaker: data.speaker, failures };
}

async function main() {
  console.log(`Generating ${COUNT} lines from ${BASE_URL} ...\n`);
  const results = [];
  for (let i = 0; i < COUNT; i++) {
    const r = await runOne();
    results.push(r);
    console.log(`[${r.failures.length === 0 ? "PASS" : "FAIL"}] (${r.speaker ?? "?"})\n${r.text}\n`);
    for (const f of r.failures) console.log(`      - ${f}`);
  }

  const failed = results.filter((r) => r.failures.length > 0).length;
  const unique = new Set(results.map((r) => r.text)).size;
  const uniqueRate = unique / results.length;

  console.log(`\n${results.length - failed}/${results.length} passed the format checks.`);
  console.log(`${unique}/${results.length} unique outputs.`);

  let bad = false;
  if (failed / results.length > MAX_FAIL_RATE) {
    console.log(`FAIL: more than ${MAX_FAIL_RATE * 100}% of outputs failed the format checks.`);
    bad = true;
  }
  if (uniqueRate < MIN_UNIQUE_RATE) {
    console.log(`FAIL: fewer than ${MIN_UNIQUE_RATE * 100}% of outputs are unique.`);
    bad = true;
  }
  console.log("\nNow read the samples above: do they sound like Shakespeare? Automated checks can't judge that.");
  if (bad) process.exitCode = 1;
}

main().catch((err) => {
  console.error("Test run crashed:", err);
  process.exitCode = 1;
});
