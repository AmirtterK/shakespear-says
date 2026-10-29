import { pipeline, env } from "@xenova/transformers";
import path from "path";
import fs from "fs";

export interface ModelStatus {
  status: "idle" | "loading" | "ready" | "error";
  progress: number;
  stage: string;
  detail?: string;
  bytesLoaded?: number;
  bytesTotal?: number;
  durationMs?: number;
  error?: string;
}

// Global singleton to persist across Next.js dev server hot-reloads
interface GlobalModelState {
  generatorPromise?: Promise<any>;
  generatorInstance?: any;
  status: ModelStatus;
  subscribers: Set<(status: ModelStatus) => void>;
  ticker?: NodeJS.Timeout;
}

const globalForModel = globalThis as unknown as { __shakespeareState?: GlobalModelState };

if (!globalForModel.__shakespeareState) {
  globalForModel.__shakespeareState = {
    status: {
      status: "idle",
      progress: 0,
      stage: "Ready to load",
      detail: "Model weights located on local disk (244 MB)",
    },
    subscribers: new Set(),
  };
}

const state = globalForModel.__shakespeareState;

function notifySubscribers() {
  const current = { ...state.status };
  for (const subscriber of state.subscribers) {
    try {
      subscriber(current);
    } catch (e) {
      console.error("[model-manager] error notifying subscriber:", e);
    }
  }
}

function updateStatus(patch: Partial<ModelStatus>) {
  state.status = { ...state.status, ...patch };
  notifySubscribers();
}

/**
 * Ensures tokenizer.json has string-formatted merges for @xenova/transformers v2.
 * Newer huggingface tokenizers save merges as [ ["Ġ", "t"], ... ] which causes
 * x.split is not a function in Xenova Transformers.
 */
function fixTokenizerIfNeeded(modelDir: string) {
  try {
    const tokenizerPath = path.join(modelDir, "tokenizer.json");
    if (fs.existsSync(tokenizerPath)) {
      const raw = fs.readFileSync(tokenizerPath, "utf-8");
      const data = JSON.parse(raw);
      if (
        data.model &&
        Array.isArray(data.model.merges) &&
        data.model.merges.length > 0 &&
        Array.isArray(data.model.merges[0])
      ) {
        console.log("[model-manager] Normalizing tokenizer.json merges array format...");
        data.model.merges = data.model.merges.map((item: any) =>
          Array.isArray(item) ? item.join(" ") : item
        );
        fs.writeFileSync(tokenizerPath, JSON.stringify(data, null, 2), "utf-8");
        console.log("[model-manager] tokenizer.json normalized successfully.");
      }
    }
  } catch (err) {
    console.warn("[model-manager] Warning: Could not verify tokenizer.json:", err);
  }
}

export function getModelStatus(): ModelStatus {
  return { ...state.status };
}

export function subscribeToModelStatus(callback: (status: ModelStatus) => void): () => void {
  state.subscribers.add(callback);
  // Send current status immediately
  try {
    callback({ ...state.status });
  } catch {}

  // If status is idle, auto-trigger loading so caller doesn't have to trigger it separately
  if (state.status.status === "idle") {
    loadModel().catch(() => {});
  }

  return () => {
    state.subscribers.delete(callback);
  };
}

export async function loadModel(): Promise<any> {
  if (state.generatorInstance) {
    updateStatus({
      status: "ready",
      progress: 100,
      stage: "Shakespeare is ready to speak!",
      detail: "Loaded in memory",
    });
    return state.generatorInstance;
  }

  if (state.generatorPromise) {
    return state.generatorPromise;
  }

  const startTime = Date.now();
  const localModelDir = path.join(process.cwd(), "src", "lib");
  const specificModelDir = path.join(localModelDir, "shakespeare-gpt2-onnx");

  fixTokenizerIfNeeded(specificModelDir);

  env.allowRemoteModels = false;
  env.localModelPath = localModelDir;

  updateStatus({
    status: "loading",
    progress: 5,
    stage: "Preparing Shakespeare model...",
    detail: "Verifying local quantized ONNX weights",
    bytesLoaded: 0,
    bytesTotal: 244726921,
  });

  // Smooth ticker for the ONNX Runtime session compilation step
  // which takes ~2-3 seconds after reading files
  let currentProgress = 5;

  const ticker = setInterval(() => {
    if (state.status.status === "loading") {
      if (currentProgress >= 65 && currentProgress < 92) {
        currentProgress += 3;
        updateStatus({
          progress: currentProgress,
          stage: "Compiling neural graph in ONNX Runtime...",
          detail: "Optimizing execution tensors & graph nodes",
        });
      }
    }
  }, 350);

  state.ticker = ticker;

  state.generatorPromise = (async () => {
    try {
      const generator = await pipeline("text-generation", "shakespeare-gpt2-onnx", {
        quantized: true,
        progress_callback: (item: any) => {
          if (!item) return;

          const file = item.file || "";
          if (file.includes("tokenizer") || file.includes("config")) {
            currentProgress = Math.max(currentProgress, 25);
            updateStatus({
              progress: currentProgress,
              stage: "Reading vocabulary & configs...",
              detail: file,
            });
          } else if (file.includes("onnx")) {
            currentProgress = Math.max(currentProgress, 60);
            updateStatus({
              progress: currentProgress,
              stage: "Loading ONNX model weights (244 MB)...",
              detail: `${item.loaded ? (item.loaded / (1024 * 1024)).toFixed(1) : "244"} MB read`,
              bytesLoaded: item.loaded || 244726921,
              bytesTotal: item.total || 244726921,
            });
          }
        },
      });

      clearInterval(ticker);

      // Warm up the session with a single token generation so first user interaction is instant
      currentProgress = 95;
      updateStatus({
        progress: 95,
        stage: "Warming up Shakespeare engine...",
        detail: "Allocating inference memory buffers",
      });

      try {
        await generator("HAMLET:\n", { max_new_tokens: 1 });
      } catch (warmupErr) {
        console.warn("[model-manager] Warmup skipped:", warmupErr);
      }

      const elapsed = Date.now() - startTime;
      state.generatorInstance = generator;

      updateStatus({
        status: "ready",
        progress: 100,
        stage: "Shakespeare is ready to speak!",
        detail: `Loaded & verified in ${(elapsed / 1000).toFixed(1)}s`,
        durationMs: elapsed,
      });

      console.log(`[model-manager] Model successfully loaded and warmed up in ${elapsed}ms`);
      return generator;
    } catch (err: any) {
      clearInterval(ticker);
      const errMsg = err instanceof Error ? err.message : String(err);
      console.error("[model-manager] Failed to load model:", errMsg);

      state.generatorPromise = undefined;
      state.generatorInstance = undefined;

      updateStatus({
        status: "error",
        progress: 0,
        stage: "Model loading failed",
        error: errMsg,
        detail: "Please check console logs or restart dev server.",
      });

      throw err;
    }
  })();

  return state.generatorPromise;
}

export async function getModel(): Promise<any> {
  return loadModel();
}
