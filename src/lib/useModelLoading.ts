"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { ModelStatus } from "./model-manager";

export function useModelLoading() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "idle",
    progress: 0,
    stage: "Connecting to model...",
  });
  const [isReady, setIsReady] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const applyStatus = useCallback((data: ModelStatus) => {
    setModelStatus((prev) => ({
      ...data,
      progress: Math.max(prev.progress, data.progress),
    }));

    if (data.status === "ready") {
      setIsReady(true);
    }
  }, []);

  const startPollingFallback = useCallback(() => {
    if (pollingTimerRef.current) return;

    const poll = async () => {
      try {
        const res = await fetch("/api/model/status", { cache: "no-store" });
        if (res.ok) {
          const data: ModelStatus = await res.json();
          applyStatus(data);
          if (data.status === "ready" || data.status === "error") {
            if (pollingTimerRef.current) {
              clearInterval(pollingTimerRef.current);
              pollingTimerRef.current = null;
            }
          }
        }
      } catch {}
    };

    poll();
    pollingTimerRef.current = setInterval(poll, 400);
  }, [applyStatus]);

  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const es = new EventSource("/api/model/status?stream=true");
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data: ModelStatus = JSON.parse(event.data);
          applyStatus(data);
          if (data.status === "ready" || data.status === "error") {
            es.close();
            eventSourceRef.current = null;
          }
        } catch (err) {
          console.error("[useModelLoading] parse error:", err);
        }
      };

      es.onerror = () => {
        es.close();
        eventSourceRef.current = null;
        startPollingFallback();
      };
    } catch {
      startPollingFallback();
    }
  }, [applyStatus, startPollingFallback]);

  const retry = useCallback(async () => {
    setIsReady(false);
    setModelStatus({
      status: "loading",
      progress: 5,
      stage: "Retrying model load...",
    });

    try {
      await fetch("/api/model/status", { method: "POST" });
    } catch {}

    connectSSE();
  }, [connectSSE]);

  useEffect(() => {
    connectSSE();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }
    };
  }, [connectSSE]);

  return {
    ...modelStatus,
    isReady,
    retry,
  };
}
