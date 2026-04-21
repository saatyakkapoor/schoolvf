/**
 * CameraStream — live video for a configured camera.
 *
 * - RTSP sources: MJPEG from GET /api/cameras/:id/stream (multipart replace).
 * - Webcam sources (stream_url webcam:N): opens the Nth local camera in the
 *   browser via getUserMedia (the API proxy is RTSP-only; plate detection still
 *   uses the vision worker on the PC).
 */
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import RefreshIcon from "@mui/icons-material/Refresh";
import VideocamOffIcon from "@mui/icons-material/VideocamOff";
import {
  Box,
  CircularProgress,
  IconButton,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { useAuth } from "../context/AuthContext";

import WebcamBrowserPreview, { parseWebcamDeviceIndex } from "./WebcamBrowserPreview";

interface CameraStreamProps {
  cameraId: string;
  cameraName?: string;
  /** DB `stream_url` — when `webcam:N`, uses browser camera preview (API MJPEG is RTSP-only). */
  cameraStreamUrl?: string;
  height?: number;
  showOverlay?: boolean;
}

type StreamState = "loading" | "streaming" | "error" | "retrying";

// If no new image load event fires within this many ms, force a src refresh.
const WATCHDOG_MS = 12_000;
// Retry delays: 1.5s, 3s, 6s, 12s (capped)
const RETRY_DELAYS_MS = [1_500, 3_000, 6_000, 12_000];

export default function CameraStream(props: CameraStreamProps) {
  const { cameraStreamUrl, cameraId, cameraName, height = 360, showOverlay = true } = props;
  const webcamIdx = cameraStreamUrl ? parseWebcamDeviceIndex(cameraStreamUrl) : null;
  if (webcamIdx !== null) {
    return (
      <WebcamBrowserPreview deviceIndex={webcamIdx} cameraName={cameraName} height={height} />
    );
  }
  return (
    <MjpegCameraStream
      cameraId={cameraId}
      cameraName={cameraName}
      height={height}
      showOverlay={showOverlay}
    />
  );
}

function MjpegCameraStream({
  cameraId,
  cameraName,
  height = 360,
  showOverlay = true,
}: Omit<CameraStreamProps, "cameraStreamUrl">) {
  const { token } = useAuth();
  const imgRef = useRef<HTMLImageElement>(null);
  const [state, setState] = useState<StreamState>("loading");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const watchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bust the browser cache on each retry attempt
  const [cacheBuster, setCacheBuster] = useState(0);

  const baseUrl = import.meta.env.VITE_API_BASE ?? "/api";

  const mjpegSrc = token
    ? `${baseUrl}/cameras/${encodeURIComponent(cameraId)}/stream?token=${encodeURIComponent(token)}${cacheBuster ? `&_t=${cacheBuster}` : ""}`
    : null;

  /** Clear both timers. */
  const clearTimers = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    if (watchdogRef.current) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
  }, []);

  /** Kick the watchdog — reset it on every successful frame. */
  const resetWatchdog = useCallback(() => {
    if (watchdogRef.current) clearTimeout(watchdogRef.current);
    watchdogRef.current = setTimeout(() => {
      // Browser hasn't fired onLoad in WATCHDOG_MS — force src refresh
      if (state === "streaming") {
        retryCountRef.current = 0; // reset backoff for watchdog triggers
        setCacheBuster(Date.now());
        setState("loading");
      }
    }, WATCHDOG_MS);
  }, [state]);

  /** Schedule a retry after exponential backoff. */
  const scheduleRetry = useCallback(() => {
    const attempt = retryCountRef.current;
    const delay = RETRY_DELAYS_MS[Math.min(attempt, RETRY_DELAYS_MS.length - 1)];
    retryCountRef.current = attempt + 1;
    setState("retrying");

    retryTimerRef.current = setTimeout(() => {
      setState("loading");
      setCacheBuster(Date.now());
    }, delay);
  }, []);

  /** Manual retry resets the backoff counter. */
  const manualRetry = useCallback(() => {
    clearTimers();
    retryCountRef.current = 0;
    setState("loading");
    setCacheBuster(Date.now());
  }, [clearTimers]);

  // Reset on camera change
  useEffect(() => {
    clearTimers();
    retryCountRef.current = 0;
    setState("loading");
    setCacheBuster(0);
    return clearTimers;
  }, [cameraId, clearTimers]);

  const handleLoad = useCallback(() => {
    setState("streaming");
    retryCountRef.current = 0;
    resetWatchdog();
  }, [resetWatchdog]);

  const handleError = useCallback(() => {
    clearTimers();
    // If we've never successfully loaded, show error immediately after first retry fails
    if (retryCountRef.current >= RETRY_DELAYS_MS.length) {
      setState("error");
    } else {
      scheduleRetry();
    }
  }, [clearTimers, scheduleRetry]);

  // Cleanup on unmount
  useEffect(() => () => clearTimers(), [clearTimers]);

  const toggleFullscreen = useCallback(() => {
    const el = imgRef.current?.closest("[data-stream-container]") as HTMLElement | null;
    if (!el) return;
    if (!document.fullscreenElement) {
      void el.requestFullscreen();
      setIsFullscreen(true);
    } else {
      void document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  if (!token || !mjpegSrc) {
    return (
      <Placeholder height={height} icon={<VideocamOffIcon sx={{ fontSize: 40, opacity: 0.3 }} />}>
        Not authenticated
      </Placeholder>
    );
  }

  return (
    <Box
      data-stream-container
      sx={{
        position: "relative",
        width: "100%",
        height,
        bgcolor: "#0a0c10",
        borderRadius: 2,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {/* Loading spinner */}
      {state === "loading" && (
        <StateOverlay>
          <CircularProgress size={32} sx={{ color: "grey.600" }} />
          <Typography variant="caption" color="grey.500" sx={{ mt: 1 }}>
            {cameraName ?? cameraId}
          </Typography>
        </StateOverlay>
      )}

      {/* Retrying state */}
      {state === "retrying" && (
        <StateOverlay>
          <CircularProgress size={28} sx={{ color: "warning.dark" }} />
          <Typography variant="caption" color="warning.dark" sx={{ mt: 1 }}>
            Reconnecting… (attempt {retryCountRef.current})
          </Typography>
          <Tooltip title="Retry now">
            <IconButton onClick={manualRetry} size="small" sx={{ mt: 0.5 }}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </StateOverlay>
      )}

      {/* Error state */}
      {state === "error" && (
        <StateOverlay>
          <ErrorOutlineIcon sx={{ fontSize: 36, color: "error.dark", opacity: 0.7 }} />
          <Typography variant="body2" color="error.light" textAlign="center" sx={{ mt: 1, px: 2 }}>
            Stream unavailable
          </Typography>
          <Typography variant="caption" color="grey.600" textAlign="center" sx={{ px: 3 }}>
            Check the RTSP URL in Camera settings and ensure the camera is reachable.
          </Typography>
          <Tooltip title="Retry">
            <IconButton onClick={manualRetry} size="small" sx={{ mt: 1 }}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </StateOverlay>
      )}

      {/* The actual MJPEG stream */}
      <Box
        component="img"
        ref={imgRef}
        src={mjpegSrc}
        alt={`Live: ${cameraName ?? cameraId}`}
        onLoad={handleLoad}
        onError={handleError}
        sx={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          display: state === "error" || state === "retrying" ? "none" : "block",
          opacity: state === "streaming" ? 1 : 0,
          transition: "opacity 0.25s ease",
        }}
      />

      {/* Overlay — camera name + live dot + controls */}
      {showOverlay && state === "streaming" && (
        <Box
          sx={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            px: 1.5,
            py: 0.75,
            background: "linear-gradient(transparent, rgba(0,0,0,0.75))",
            display: "flex",
            alignItems: "center",
            gap: 1,
          }}
        >
          {/* Pulsing live dot */}
          <Box
            sx={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              bgcolor: "error.main",
              flexShrink: 0,
              "@keyframes livePulse": {
                "0%, 100%": { opacity: 1, transform: "scale(1)" },
                "50%": { opacity: 0.35, transform: "scale(0.8)" },
              },
              animation: "livePulse 1.6s ease-in-out infinite",
            }}
          />
          <Typography
            variant="caption"
            sx={{ color: "rgba(255,255,255,0.88)", fontWeight: 600, flex: 1 }}
          >
            {cameraName ?? cameraId}
          </Typography>
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.4)", fontSize: 10 }}>
            LIVE
          </Typography>
          <Tooltip title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}>
            <IconButton
              size="small"
              onClick={toggleFullscreen}
              sx={{ color: "rgba(255,255,255,0.5)", p: 0.25, "&:hover": { color: "white" } }}
            >
              {isFullscreen ? (
                <FullscreenExitIcon sx={{ fontSize: 16 }} />
              ) : (
                <FullscreenIcon sx={{ fontSize: 16 }} />
              )}
            </IconButton>
          </Tooltip>
        </Box>
      )}
    </Box>
  );
}

// ── helpers ─────────────────────────────────────────────────────────────────

function Placeholder({
  height,
  icon,
  children,
}: {
  height: number;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Box
      sx={{
        width: "100%",
        height,
        bgcolor: "#0a0c10",
        borderRadius: 2,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 1,
        color: "grey.700",
        border: "1px solid rgba(255,255,255,0.04)",
      }}
    >
      {icon}
      <Typography variant="caption">{children}</Typography>
    </Box>
  );
}

function StateOverlay({ children }: { children: React.ReactNode }) {
  return (
    <Box
      sx={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 2,
        bgcolor: "rgba(10,12,16,0.9)",
      }}
    >
      {children}
    </Box>
  );
}
