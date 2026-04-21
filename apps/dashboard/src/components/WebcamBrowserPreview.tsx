/**
 * Local webcam preview via getUserMedia — matches OpenCV index order (0, 1, … videoinput devices).
 * Used when the camera is configured as webcam:N because the API MJPEG proxy only supports RTSP.
 */
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { Box, CircularProgress, Typography } from "@mui/material";
import { useCallback, useEffect, useRef, useState } from "react";

export function parseWebcamDeviceIndex(streamUrl: string): number | null {
  const m = streamUrl.trim().toLowerCase().match(/^webcam:(\d+)$/);
  if (!m) return null;
  return parseInt(m[1], 10);
}

type PreviewState = "loading" | "live" | "error";

interface WebcamBrowserPreviewProps {
  deviceIndex: number;
  cameraName?: string;
  height?: number;
}

export default function WebcamBrowserPreview({
  deviceIndex,
  cameraName,
  height = 360,
}: WebcamBrowserPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<PreviewState>("loading");
  const [message, setMessage] = useState<string>("");

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setState("error");
        setMessage("This browser does not support camera access.");
        return;
      }

      setState("loading");
      setMessage("");

      try {
        const pre = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        pre.getTracks().forEach((t) => t.stop());
      } catch {
        // permission may still fail on exact device below
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter((d) => d.kind === "videoinput");
      if (cancelled) return;

      if (deviceIndex < 0 || deviceIndex >= inputs.length) {
        setState("error");
        setMessage(
          `No camera at index ${deviceIndex}. This browser sees ${inputs.length} video input(s). ` +
            "Pick another index on the Cameras page or run scripts/list_opencv_cameras.py on the vision worker PC.",
        );
        return;
      }

      const chosen = inputs[deviceIndex];
      if (!chosen?.deviceId) {
        setState("error");
        setMessage("Could not read device id — allow camera permission and refresh.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: chosen.deviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          await v.play().catch(() => undefined);
        }
        setState("live");
      } catch (e) {
        setState("error");
        setMessage(
          e instanceof Error
            ? e.message
            : "Could not open the camera. Check browser permissions and that no other app is using it.",
        );
      }
    }

    void run();
    return () => {
      cancelled = true;
      stopTracks();
    };
  }, [deviceIndex, stopTracks]);

  if (state === "error") {
    return (
      <Box
        sx={{
          width: "100%",
          height,
          bgcolor: "#0a0c10",
          borderRadius: 2,
          border: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 1,
          px: 2,
        }}
      >
        <ErrorOutlineIcon sx={{ fontSize: 36, color: "error.dark", opacity: 0.8 }} />
        <Typography variant="body2" color="error.light" textAlign="center">
          Webcam preview unavailable
        </Typography>
        <Typography variant="caption" color="grey.500" textAlign="center" sx={{ maxWidth: 420 }}>
          {message}
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        height,
        bgcolor: "#0a0c10",
        borderRadius: 2,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.06)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {state === "loading" && (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2,
            bgcolor: "rgba(10,12,16,0.92)",
            gap: 1,
          }}
        >
          <CircularProgress size={32} sx={{ color: "grey.600" }} />
          <Typography variant="caption" color="grey.500">
            Opening camera {deviceIndex}…
          </Typography>
        </Box>
      )}
      <Box
        component="video"
        ref={videoRef}
        autoPlay
        playsInline
        muted
        sx={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          display: "block",
          opacity: state === "live" ? 1 : 0,
        }}
      />
      {state === "live" && (
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
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.88)", fontWeight: 600, flex: 1 }}>
            {cameraName ?? `Camera ${deviceIndex}`}
          </Typography>
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.45)", fontSize: 10 }}>
            Browser preview · index {deviceIndex}
          </Typography>
        </Box>
      )}
    </Box>
  );
}
