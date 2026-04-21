/**
 * LivePage — split panel: left = MJPEG camera stream, right = real-time plate feed.
 *
 * - Camera selector tabs across the top (one tab per configured camera).
 * - Left: RTSP cameras use MJPEG from /api/cameras/:id/stream; webcam:N uses the browser camera.
 * - Right: WebSocket plate detection feed for all cameras (entry + exit together).
 * - No mock data — RTSP errors surface from the API; webcams use in-browser preview (API is RTSP-only).
 */
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  Grid,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { getCameras } from "../api/cameras";
import { getLiveDebug } from "../api/live";
import CameraStream from "../components/CameraStream";
import LiveDetectionFeed from "../components/LiveDetectionFeed";
import { useLiveWebSocket } from "../hooks/useLiveWebSocket";
import type { LiveDebugEntry } from "../types";
import { getApiErrorMessage } from "../utils/errors";

const RAW_LINE_MAX = 2800;

function trimRawLine(line: string): string {
  if (line.length <= RAW_LINE_MAX) return line;
  return `${line.slice(0, RAW_LINE_MAX)} … [truncated]`;
}

export default function LivePage() {
  const { connected, rawWsLog, debugEntries: wsDebugEntries } = useLiveWebSocket();

  const {
    data: restDebug = [],
    isError: liveDebugQueryFailed,
    error: liveDebugError,
  } = useQuery({
    queryKey: ["live", "debug", 200],
    queryFn: () => getLiveDebug(200),
    refetchInterval: 4_000,
  });

  const mergedDebug = useMemo(() => {
    const byId = new Map<string, LiveDebugEntry>();
    for (const e of restDebug) {
      if (e?.id) byId.set(e.id, e);
    }
    for (const e of wsDebugEntries) {
      if (e?.id) byId.set(e.id, e);
    }
    return Array.from(byId.values())
      .sort((a, b) => (a.at < b.at ? 1 : a.at > b.at ? -1 : 0))
      .slice(0, 120);
  }, [restDebug, wsDebugEntries]);

  const {
    data: cameras,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ["cameras"],
    queryFn: getCameras,
    staleTime: 30_000,
  });

  const [selectedIdx, setSelectedIdx] = useState(0);
  const selectedCamera = cameras?.[selectedIdx] ?? null;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {/* Page header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, flexWrap: "wrap" }}>
        <CameraAltIcon sx={{ color: "text.secondary" }} />
        <Typography variant="h5" fontWeight={600}>
          Live Monitor
        </Typography>
        {/* WS badge */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            px: 1,
            py: 0.25,
            borderRadius: 10,
            bgcolor: connected ? "success.dark" : "action.disabledBackground",
            border: "1px solid",
            borderColor: connected ? "success.main" : "divider",
          }}
        >
          <Box
            sx={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              bgcolor: connected ? "success.light" : "text.disabled",
              "@keyframes blink": { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0.2 } },
              animation: connected ? "blink 1.4s ease-in-out infinite" : "none",
            }}
          />
          <Typography variant="caption" sx={{ color: connected ? "success.light" : "text.disabled" }}>
            {connected ? "WebSocket live" : "Connecting…"}
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ ml: "auto" }}>
          {selectedCamera?.stream_url?.toLowerCase().startsWith("webcam:")
            ? "Video: local webcam in the browser."
            : "Video: API → MJPEG in the browser."}
        </Typography>
      </Box>

      {/* Camera tabs */}
      {isLoading ? (
        <Box display="flex" alignItems="center" gap={1}>
          <CircularProgress size={16} />
          <Typography variant="body2" color="text.secondary">
            Loading cameras…
          </Typography>
        </Box>
      ) : isError ? (
        <Alert severity="error">{getApiErrorMessage(error)}</Alert>
      ) : !cameras || cameras.length === 0 ? (
        <Alert severity="warning">
          No cameras configured. Add cameras in the Cameras page first.
        </Alert>
      ) : (
        <>
          <Tabs
            value={selectedIdx}
            onChange={(_, v: number) => setSelectedIdx(v)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              borderBottom: "1px solid",
              borderColor: "divider",
              minHeight: 40,
              "& .MuiTab-root": { minHeight: 40, textTransform: "none", fontSize: 13 },
            }}
          >
            {cameras.map((cam, idx) => (
              <Tab
                key={cam.id}
                label={
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                    <Box
                      sx={{
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        bgcolor:
                          cam.status === "online"
                            ? "success.main"
                            : cam.status === "error"
                            ? "error.main"
                            : "warning.main",
                        flexShrink: 0,
                      }}
                    />
                    {cam.name}
                    <Typography
                      component="span"
                      variant="caption"
                      sx={{
                        bgcolor: "action.hover",
                        px: 0.75,
                        py: 0.1,
                        borderRadius: 1,
                        fontSize: 10,
                        color: "text.secondary",
                      }}
                    >
                      {cam.gate_type.toUpperCase()}
                    </Typography>
                  </Box>
                }
                value={idx}
              />
            ))}
          </Tabs>

          {selectedCamera && (
            <Grid container spacing={2} sx={{ flexGrow: 1 }}>
              {/* ── Left: camera stream ───────────────────────────── */}
              <Grid item xs={12} md={7}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                  <CameraStream
                    cameraId={selectedCamera.id}
                    cameraName={selectedCamera.name}
                    cameraStreamUrl={selectedCamera.stream_url}
                    height={420}
                    showOverlay
                  />
                  <Typography variant="caption" color="text.disabled">
                    {selectedCamera.stream_url.toLowerCase().startsWith("webcam:")
                      ? "Preview uses your browser. Plate detection uses the vision worker on this PC with the same OpenCV index."
                      : "Switch tabs for the other angle. RTSP URLs are edited on the Cameras page."}
                  </Typography>
                </Box>
              </Grid>

              {/* ── Right: live plate detection feed (all cameras — not filtered by tab) ─ */}
              <Grid item xs={12} md={5}>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Detections · all cameras
                  </Typography>
                  <LiveDetectionFeed height={480} maxItems={80} />
                  <Typography variant="caption" color="text.disabled">
                    Entry and exit are both listed here. Tabs only change the video on the left.
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          )}
        </>
      )}

      {/* Pipeline / WebSocket debug — always visible on this page */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Debug log
        </Typography>
        <Typography variant="caption" color="text.disabled" display="block" sx={{ mb: 1 }}>
          Raw WebSocket payloads (exactly as received) and structured events from GET /live/debug
          (API ingest, vision worker, merged with the socket).
        </Typography>
        {liveDebugQueryFailed && (
          <Alert severity="error" sx={{ mb: 1 }}>
            GET /live/debug failed: {getApiErrorMessage(liveDebugError)} — REST debug will stay empty; check
            auth and API.
          </Alert>
        )}
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="caption" fontWeight={600} color="text.secondary">
              Raw WebSocket ({rawWsLog.length} lines)
            </Typography>
            <Box
              component="pre"
              sx={{
                mt: 0.5,
                maxHeight: 320,
                overflow: "auto",
                m: 0,
                p: 1.5,
                borderRadius: 1,
                bgcolor: "grey.900",
                color: "grey.100",
                fontSize: 11,
                lineHeight: 1.35,
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
                border: "1px solid",
                borderColor: "divider",
              }}
            >
              {rawWsLog.length === 0
                ? connected
                  ? "Waiting for messages…"
                  : "Connect with a valid session to open the WebSocket."
                : rawWsLog.map((line, i) => (
                    <span key={`${i}-${line.slice(0, 24)}`}>
                      {trimRawLine(line)}
                      {"\n"}
                    </span>
                  ))}
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="caption" fontWeight={600} color="text.secondary">
              Structured debug ({mergedDebug.length} events)
            </Typography>
            <Box
              sx={{
                mt: 0.5,
                maxHeight: 320,
                overflow: "auto",
                p: 1,
                borderRadius: 1,
                bgcolor: "action.hover",
                border: "1px solid",
                borderColor: "divider",
                fontFamily: "monospace",
                fontSize: 11,
              }}
            >
              {mergedDebug.length === 0 ? (
                <Typography variant="caption" color="text.disabled">
                  No debug rows yet. Vision worker pushes /api/live/debug; plate ingest also logs
                  here.
                </Typography>
              ) : (
                mergedDebug.map((e) => (
                  <Box
                    key={e.id}
                    sx={{
                      py: 0.75,
                      borderBottom: "1px solid",
                      borderColor: "divider",
                      "&:last-child": { borderBottom: "none" },
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
                      <Typography variant="caption" sx={{ color: "text.secondary", fontFamily: "inherit" }}>
                        {e.at}
                      </Typography>
                      <Typography variant="caption" fontWeight={600} sx={{ fontFamily: "inherit" }}>
                        {e.message}
                      </Typography>
                      <Chip label={e.source} size="small" variant="outlined" sx={{ height: 18, fontSize: 10 }} />
                    </Box>
                    <Box
                      component="pre"
                      sx={{
                        m: 0,
                        mt: 0.5,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        color: "text.secondary",
                        fontFamily: "inherit",
                        fontSize: 10,
                      }}
                    >
                      {JSON.stringify(e.detail ?? {}, null, 2)}
                    </Box>
                  </Box>
                ))
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}
