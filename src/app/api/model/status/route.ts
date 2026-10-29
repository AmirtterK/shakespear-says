import { NextRequest } from "next/server";
import {
  getModelStatus,
  subscribeToModelStatus,
  loadModel,
  type ModelStatus,
} from "@/lib/model-manager";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const wantsStream =
    searchParams.get("stream") === "true" ||
    req.headers.get("accept")?.includes("text/event-stream");

  // If streaming is requested (Server-Sent Events)
  if (wantsStream) {
    const encoder = new TextEncoder();
    let unsubscribe: (() => void) | null = null;
    let isClosed = false;

    const stream = new ReadableStream({
      start(controller) {
        function sendEvent(data: ModelStatus) {
          if (isClosed) return;
          try {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
          } catch {
            isClosed = true;
          }
        }

        unsubscribe = subscribeToModelStatus((status) => {
          sendEvent(status);
          if (status.status === "ready" || status.status === "error") {
            // Keep connection briefly to ensure delivery then close gracefully
            setTimeout(() => {
              if (!isClosed) {
                isClosed = true;
                try {
                  controller.close();
                } catch {}
              }
            }, 300);
          }
        });
      },
      cancel() {
        isClosed = true;
        if (unsubscribe) {
          unsubscribe();
          unsubscribe = null;
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform, no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  }

  // Non-streaming polling fallback
  const current = getModelStatus();
  if (current.status === "idle") {
    loadModel().catch(() => {});
  }

  return Response.json(getModelStatus());
}

export async function POST() {
  loadModel().catch(() => {});
  return Response.json(getModelStatus());
}
