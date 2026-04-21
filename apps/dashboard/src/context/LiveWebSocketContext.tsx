import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { useAuth } from "./AuthContext";
import type { LiveDebugEntry, LiveDetection } from "../types";
import { buildLiveWebSocketUrl, redactWebSocketUrlForLog } from "../utils/buildLiveWebSocketUrl";
import { playPlateDetectedSound } from "../utils/plateDetectionSound";

type WsMessage =
  | { type: "snapshot"; recent: LiveDetection[]; debug?: LiveDebugEntry[] }
  | { type: "detection"; payload: LiveDetection }
  | { type: "debug"; payload: LiveDebugEntry }
  | { type: "detection_adjusted"; event_id: string; swap_type: string; swap_notes: string | null; swap_resolved: boolean; swap_resolved_by: string };

const RAW_LOG_MAX = 120;
const DEBUG_MAX = 200;

function parseMessage(raw: string): WsMessage | null {
  try {
    const msg = JSON.parse(raw) as Record<string, unknown>;
    if (msg.type === "snapshot" && Array.isArray(msg.recent)) {
      return {
        type: "snapshot",
        recent: msg.recent as LiveDetection[],
        debug: Array.isArray(msg.debug) ? (msg.debug as LiveDebugEntry[]) : undefined,
      };
    }
    if (msg.type === "detection" && msg.payload && typeof msg.payload === "object") {
      return { type: "detection", payload: msg.payload as LiveDetection };
    }
    if (msg.type === "debug" && msg.payload && typeof msg.payload === "object") {
      return { type: "debug", payload: msg.payload as LiveDebugEntry };
    }
    if (msg.type === "detection_adjusted" && typeof msg.event_id === "string") {
      return {
        type: "detection_adjusted",
        event_id: msg.event_id as string,
        swap_type: msg.swap_type as string,
        swap_notes: (msg.swap_notes as string | null) ?? null,
        swap_resolved: msg.swap_resolved as boolean,
        swap_resolved_by: msg.swap_resolved_by as string,
      };
    }
  } catch {
    /* ignore */
  }
  return null;
}

export type LiveWebSocketValue = {
  connected: boolean;
  recent: LiveDetection[];
  snapshotReceived: boolean;
  rawWsLog: string[];
  debugEntries: LiveDebugEntry[];
};

const LiveWebSocketContext = createContext<LiveWebSocketValue | null>(null);

export function LiveWebSocketProvider({ children }: { children: ReactNode }) {
  const { token } = useAuth();
  const [connected, setConnected] = useState(false);
  const [recent, setRecent] = useState<LiveDetection[]>([]);
  const [snapshotReceived, setSnapshotReceived] = useState(false);
  const [rawWsLog, setRawWsLog] = useState<string[]>([]);
  const [debugEntries, setDebugEntries] = useState<LiveDebugEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const applyMessage = useCallback((msg: WsMessage) => {
    if (msg.type === "snapshot") {
      setRecent(msg.recent);
      if (msg.debug !== undefined) {
        setDebugEntries(msg.debug);
      }
      setSnapshotReceived(true);
      return;
    }
    if (msg.type === "debug") {
      setDebugEntries((prev) => [msg.payload, ...prev].slice(0, DEBUG_MAX));
      return;
    }
    if (msg.type === "detection_adjusted") {
      // Patch the matching record in-place so all feeds update immediately
      setRecent((prev) =>
        prev.map((r) =>
          r.id === msg.event_id
            ? { ...r, swap_type: msg.swap_type, swap_notes: msg.swap_notes, swap_resolved: msg.swap_resolved, swap_resolved_by: msg.swap_resolved_by }
            : r,
        ),
      );
      return;
    }
    if (msg.type === "detection") {
      playPlateDetectedSound();
    }
    setRecent((prev) => {
      const next = [msg.payload, ...prev];
      return next.slice(0, 200);
    });
  }, []);

  useEffect(() => {
    if (!token) return;

    let cleaned = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let ws: WebSocket | null = null;

    const pushDiag = (line: string) => {
      const stamp = new Date().toISOString();
      setRawWsLog((lines) => [`${stamp}  ${line}`, ...lines].slice(0, RAW_LOG_MAX));
    };

    const connect = () => {
      if (cleaned) return;
      const url = buildLiveWebSocketUrl(token);
      pushDiag(`[socket] connecting ${redactWebSocketUrlForLog(url)}`);
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cleaned) return;
        setConnected(true);
        pushDiag("[socket] open");
      };

      ws.onclose = (ev) => {
        setConnected(false);
        wsRef.current = null;
        pushDiag(`[socket] closed code=${ev.code} reason=${ev.reason || "(none)"}`);
        if (cleaned) return;
        // 4401 = FastAPI close when JWT invalid — do not spin forever
        if (ev.code !== 4401 && ev.code !== 1000) {
          reconnectTimer = setTimeout(connect, 2500);
        }
      };

      ws.onerror = () => {
        pushDiag("[socket] error (browser blocked URL, TLS mix, or upstream refused — check Network tab)");
        setConnected(false);
      };

      ws.onmessage = (ev) => {
        const stamp = new Date().toISOString();
        const data = typeof ev.data === "string" ? ev.data : String(ev.data);
        setRawWsLog((lines) => [`${stamp}  ${data}`, ...lines].slice(0, RAW_LOG_MAX));
        // Respond to server keepalive pings immediately so the connection stays alive
        try {
          const parsed = JSON.parse(data) as Record<string, unknown>;
          if (parsed.type === "ping") {
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: "pong" }));
            }
            return; // Don't pass ping frames to applyMessage
          }
        } catch {
          /* not JSON — fall through */
        }
        const msg = parseMessage(data);
        if (msg) applyMessage(msg);
      };
    };

    connect();

    return () => {
      cleaned = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close(1000, "page update");
      wsRef.current = null;
    };
  }, [token, applyMessage]);

  const value = useMemo(
    () => ({
      connected,
      recent,
      snapshotReceived,
      rawWsLog,
      debugEntries,
    }),
    [connected, recent, snapshotReceived, rawWsLog, debugEntries],
  );

  return <LiveWebSocketContext.Provider value={value}>{children}</LiveWebSocketContext.Provider>;
}

export function useLiveWebSocket(): LiveWebSocketValue {
  const ctx = useContext(LiveWebSocketContext);
  if (!ctx) {
    throw new Error("useLiveWebSocket must be used inside LiveWebSocketProvider (wrap MainLayout).");
  }
  return ctx;
}
