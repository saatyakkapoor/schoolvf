/**
 * Build the live WebSocket URL using the same API origin as axios
 * (`import.meta.env.VITE_API_BASE` / default `/api`).
 *
 * If this used only `window.location.host` while `VITE_API_BASE` pointed at
 * another host (common in dev), the socket would never reach the API.
 */
export function buildLiveWebSocketUrl(token: string): string {
  const raw = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
  const base = (raw && raw.length > 0 ? raw : "/api").replace(/\/$/, "");

  if (!base.startsWith("http://") && !base.startsWith("https://")) {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const pathPrefix = base.startsWith("/") ? base : `/${base}`;
    return `${wsProto}//${window.location.host}${pathPrefix}/ws/live?token=${encodeURIComponent(token)}`;
  }

  let url: URL;
  try {
    url = new URL(base);
  } catch {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${wsProto}//${window.location.host}/api/ws/live?token=${encodeURIComponent(token)}`;
  }

  const wsProto = url.protocol === "https:" ? "wss:" : "ws:";
  let pathname = url.pathname.replace(/\/$/, "") || "";
  if (!pathname.endsWith("/api")) {
    pathname = `${pathname}/api`.replace(/\/+/g, "/");
  }
  return `${wsProto}//${url.host}${pathname}/ws/live?token=${encodeURIComponent(token)}`;
}

export function redactWebSocketUrlForLog(url: string): string {
  return url.replace(/token=[^&]+/, "token=…");
}
