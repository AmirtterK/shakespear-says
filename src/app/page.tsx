"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";

type Result = {
  text: string;
};

export default function Home() {
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleGenerate() {
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
    <main className="min-h-screen bg-background flex flex-col">
      <Toaster position="bottom-right" />

      {/* Header */}
      <div className="border-b bg-secondary/30 py-4 px-4">
        <div className="mx-auto max-w-2xl flex items-center gap-3">
          <img src="/shakespeare.jpg" alt="Shakespeare" className="w-10 h-10 rounded-full object-cover" />
          <h1 className="text-2xl font-bold">Shakespeare Says</h1>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto px-4 py-8">
        <div className="mx-auto max-w-2xl space-y-6">
          {/* Generate Button */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <Button
              onClick={handleGenerate}
              disabled={isLoading}
              className="w-full"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate a Random Line"
              )}
            </Button>
          </motion.div>

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="rounded-lg bg-destructive/10 border border-destructive/30 p-3 text-sm text-destructive"
            >
              {error}
            </motion.div>
          )}

          {/* Result Section */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              <div className="rounded-lg bg-secondary/50 border p-4">
                <p className="text-base leading-7 text-foreground">
                  {result.text}
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </main>
  );
}
