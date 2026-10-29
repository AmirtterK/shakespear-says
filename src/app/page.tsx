"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ThinkingOrb } from "thinking-orbs";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { useModelLoading } from "@/lib/useModelLoading";

type Result = {
  text: string;
};

export default function Home() {
  const {
    progress: modelProgress,
    stage: modelStage,
    isReady: isModelReady,
    status: modelStatus,
    error: modelError,
    retry: retryModelLoad,
  } = useModelLoading();

  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleGenerate() {
    if (!isModelReady || isLoading) return;
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "Generation failed.");
      }

      setResult(data);
    } catch (generationError) {
      setError(
        generationError instanceof Error
          ? generationError.message
          : "Generation failed.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col">
      <Toaster position="bottom-right" />

      {/* Header */}
      <div className="border-b bg-secondary/30 py-4 px-4">
        <div className="mx-auto max-w-2xl flex items-center justify-between">
          <h1 className="text-xl font-bold">Shakespeare Says</h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto px-4 py-8">
        <div className="mx-auto max-w-2xl space-y-6">
          {/* Shakespeare Profile & Status Row */}
          <div className="flex items-center gap-4 p-1">
            {!isModelReady ? (
              <div className="w-16 h-16 flex items-center justify-center shrink-0">
                <ThinkingOrb state="solving" size={64} />
              </div>
            ) : isLoading ? (
              <div className="w-16 h-16 flex items-center justify-center shrink-0">
                <ThinkingOrb state="composing" size={64} />
              </div>
            ) : (
              <img
                src="/shakespeare.png"
                alt="Shakespeare"
                className="w-16 h-16 rounded-full object-cover shrink-0"
              />
            )}
            <div className="text-base">
              {!isModelReady ? (
                <span className="text-foreground">
                  {modelStage || "Loading model..."} ({modelProgress}%)
                </span>
              ) : isLoading ? (
                <span className="text-foreground">
                  The Bard is composing...
                </span>
              ) : (
                <span className="text-muted-foreground">
                  Summon an Elizabethan passage.
                </span>
              )}
            </div>
          </div>

          {/* Output / Result Section (Above the button) */}
          {result && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div className="rounded-lg bg-secondary/50 border p-4">
                <p className="text-base leading-7 text-foreground whitespace-pre-line">
                  {result.text}
                </p>
              </div>
            </motion.div>
          )}

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="rounded-lg bg-destructive/10 border border-destructive/30 p-3 text-sm text-destructive"
            >
              {error}
            </motion.div>
          )}

          {/* Loader bar on launch (above button) */}
          {!isModelReady && (
            <div className="rounded-lg border bg-secondary/40 p-3 space-y-2">
              <div className="flex justify-between items-center text-xs text-muted-foreground">
                <span>{modelStage || "Loading model..."}</span>
                <span className="font-semibold text-foreground">
                  {modelProgress}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full bg-primary transition-all duration-300 ease-out"
                  style={{ width: `${modelProgress}%` }}
                />
              </div>
              {modelStatus === "error" && (
                <div className="flex items-center justify-between pt-1 text-xs text-destructive">
                  <span>{modelError || "Failed to load model."}</span>
                  <button
                    type="button"
                    onClick={retryModelLoad}
                    className="underline font-medium hover:opacity-80"
                  >
                    Retry
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Generate Button - below the output */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Button
              onClick={handleGenerate}
              disabled={isLoading || !isModelReady}
              className="w-full"
              size="lg"
            >
              Generate
            </Button>
          </motion.div>
        </div>
      </div>
    </main>
  );
}
