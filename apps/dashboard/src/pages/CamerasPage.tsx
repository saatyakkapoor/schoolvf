import {
  Add as AddIcon,
  UsbRounded as UsbIcon,
  VideocamRounded as RtspIcon,
  RefreshRounded as RefreshIcon,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  MenuItem,
  Paper,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";

import { createCamera, getCameras, probeCamera, updateCamera } from "../api/cameras";
import { getApiErrorMessage } from "../utils/errors";
import { GateType, type Camera, type CreateCameraPayload, type UpdateCameraPayload } from "../types";

const GATE_OPTIONS: GateType[] = [GateType.EXIT, GateType.ENTRY];

type SourceType = "rtsp" | "webcam";

interface DetectedCamera {
  /** OpenCV / OS enumeration order on this machine (0, 1, …) — must match the vision worker host. */
  index: number;
  label: string;
  deviceId: string;
  /** Chromium groups lenses of the same physical device; not a USB port number (unavailable in browsers). */
  groupId: string;
}

function getWebcamIndex(url: string): number {
  const idx = parseInt(url.split(":")[1] ?? "0", 10);
  return isNaN(idx) ? 0 : idx;
}

/** Shorten long MediaDevices IDs for display; full value still visible in title tooltip via CSS/attr. */
function formatOpaqueId(id: string, head = 20, tail = 12): string {
  if (!id) return "—";
  if (id.length <= head + tail + 1) return id;
  return `${id.slice(0, head)}…${id.slice(-tail)}`;
}

function isLikelyBuiltInCamera(label: string): boolean {
  const s = label.trim().toLowerCase();
  if (!s) return false;
  if (s === "hd camera" || s === "full hd camera") return true;
  return /facetime|integrated|built-in|built in/.test(s);
}

function pickDefaultWebcamIndex(cams: DetectedCamera[]): number {
  const i = cams.findIndex((c) => isLikelyBuiltInCamera(c.label));
  if (i >= 0) return cams[i]!.index;
  return cams[0]!.index;
}

function streamLabel(url: string, detectedCams?: DetectedCamera[]) {
  if (url.toLowerCase().startsWith("webcam:")) {
    const idx = getWebcamIndex(url);
    const cam = detectedCams?.find((c) => c.index === idx);
    return cam ? `${cam.label} (OpenCV index ${idx})` : `Webcam · OpenCV index ${idx}`;
  }
  return url;
}

const EMPTY_FORM = {
  name: "",
  sourceType: "rtsp" as SourceType,
  rtspUrl: "",
  webcamIndex: 0,
  gate: GateType.EXIT,
  active: true,
};

/**
 * Enumerate video inputs after requesting access. Prefers 720p+ so built-in HD cameras come up in HD when allowed.
 * Index order is the browser/OS order — usually aligns with OpenCV on the same computer.
 */
async function detectWebcams(): Promise<DetectedCamera[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];

  const hdVideo: MediaTrackConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: hdVideo, audio: false });
    stream.getTracks().forEach((t) => t.stop());
  } catch {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stream.getTracks().forEach((t) => t.stop());
    } catch {
      // Permission denied or no camera — continue; labels may be empty
    }
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter((d) => d.kind === "videoinput");
  return videoDevices.map((d, i) => ({
    index: i,
    label: d.label || `Camera ${i}`,
    deviceId: d.deviceId,
    groupId: d.groupId || "",
  }));
}

export default function CamerasPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["cameras"],
    queryFn: getCameras,
  });

  const [dialog, setDialog] = useState<Camera | "new" | null>(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [detectedCams, setDetectedCams] = useState<DetectedCamera[]>([]);
  const [camDetecting, setCamDetecting] = useState(false);

  const runDetect = useCallback(async () => {
    setCamDetecting(true);
    try {
      const cams = await detectWebcams();
      setDetectedCams(cams);
      if (cams.length > 0) {
        setForm((f) => ({ ...f, webcamIndex: pickDefaultWebcamIndex(cams) }));
      }
    } finally {
      setCamDetecting(false);
    }
  }, []);

  // Auto-detect when switching to webcam mode
  useEffect(() => {
    if (form.sourceType === "webcam" && dialog !== null && detectedCams.length === 0) {
      void runDetect();
    }
  }, [form.sourceType, dialog, detectedCams.length, runDetect]);

  const openCreate = () => {
    setForm(EMPTY_FORM);
    setDetectedCams([]);
    setDialog("new");
  };

  const openEdit = (c: Camera) => {
    const src = c.stream_url.toLowerCase().startsWith("webcam:") ? "webcam" : "rtsp";
    setForm({
      name: c.name,
      sourceType: src,
      rtspUrl: src === "rtsp" ? c.stream_url : "",
      webcamIndex: src === "webcam" ? getWebcamIndex(c.stream_url) : 0,
      gate: c.gate_type,
      active: c.is_active,
    });
    setDetectedCams([]);
    setDialog(c);
  };

  const closeDialog = () => setDialog(null);

  const buildStreamUrl = (): string =>
    form.sourceType === "webcam" ? `webcam:${form.webcamIndex}` : form.rtspUrl.trim();

  const saveMut = useMutation({
    mutationFn: async () => {
      const stream_url = buildStreamUrl();
      if (dialog === "new") {
        const payload: CreateCameraPayload = {
          name: form.name.trim(),
          gate_type: form.gate,
          stream_url,
        };
        return createCamera(payload);
      } else {
        const payload: UpdateCameraPayload = {
          name: form.name.trim(),
          stream_url,
          gate_type: form.gate,
          is_active: form.active,
        };
        return updateCamera((dialog as Camera).id, payload);
      }
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["cameras"] });
      closeDialog();
    },
  });

  const probeMut = useMutation({
    mutationFn: (id: string) => probeCamera(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["cameras"] });
    },
  });

  const canSave =
    form.name.trim().length > 0 &&
    (form.sourceType === "webcam" || form.rtspUrl.trim().length > 0);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return <Alert severity="error">{getApiErrorMessage(error)}</Alert>;
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
        <Typography variant="h4">Cameras</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate}>
          Add Camera
        </Button>
      </Box>
      <Typography color="text.secondary" paragraph>
        Add RTSP network cameras or USB/built-in webcams. The vision worker picks up changes within
        30 seconds — <strong>no browser tab required</strong>.
      </Typography>

      {probeMut.isSuccess && probeMut.data?.hint && (
        <Alert severity="info" sx={{ mb: 2 }} onClose={() => probeMut.reset()}>
          {probeMut.data.hint}
        </Alert>
      )}
      {probeMut.isError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {getApiErrorMessage(probeMut.error)}
        </Alert>
      )}
      {saveMut.isError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {getApiErrorMessage(saveMut.error)}
        </Alert>
      )}

      <Paper variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Gate</TableCell>
              <TableCell>Source</TableCell>
              <TableCell>Active</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data!.map((c) => {
              const isWebcam = c.stream_url.toLowerCase().startsWith("webcam:");
              return (
                <TableRow key={c.id}>
                  <TableCell>{c.name}</TableCell>
                  <TableCell>{c.gate_type}</TableCell>
                  <TableCell>
                    <Chip
                      icon={isWebcam ? <UsbIcon /> : <RtspIcon />}
                      label={streamLabel(c.stream_url)}
                      size="small"
                      variant="outlined"
                      sx={{ fontFamily: "monospace", fontSize: 11 }}
                    />
                  </TableCell>
                  <TableCell>{c.is_active ? "yes" : "no"}</TableCell>
                  <TableCell>{c.status}</TableCell>
                  <TableCell align="right">
                    <Button size="small" sx={{ mr: 1 }} onClick={() => openEdit(c)}>
                      Edit
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      disabled={probeMut.isPending}
                      onClick={() => probeMut.mutate(c.id)}
                    >
                      {isWebcam ? "Check" : "Test TCP"}
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>

      {/* Add / Edit dialog */}
      <Dialog open={dialog !== null} onClose={closeDialog} fullWidth maxWidth="sm">
        <DialogTitle>
          {dialog === "new" ? "Add Camera" : `Edit — ${(dialog as Camera)?.name}`}
        </DialogTitle>
        <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 2 }}>
          <TextField
            label="Name"
            value={form.name}
            onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
            fullWidth
          />

          {/* Source type toggle */}
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
              Source type
            </Typography>
            <ToggleButtonGroup
              value={form.sourceType}
              exclusive
              onChange={(_, v) => v && setForm((f) => ({ ...f, sourceType: v as SourceType }))}
              size="small"
            >
              <ToggleButton value="rtsp" sx={{ gap: 0.5 }}>
                <RtspIcon fontSize="small" /> RTSP / IP Camera
              </ToggleButton>
              <ToggleButton value="webcam" sx={{ gap: 0.5 }}>
                <UsbIcon fontSize="small" /> Webcam / Built-in
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {form.sourceType === "rtsp" ? (
            <TextField
              label="RTSP URL"
              value={form.rtspUrl}
              onChange={(e) => setForm((f) => ({ ...f, rtspUrl: e.target.value }))}
              fullWidth
              multiline
              minRows={2}
              helperText='e.g. rtsp://admin:pass%401234@192.168.1.12:554/Streaming/Channels/101  (encode @ as %40)'
            />
          ) : (
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="caption" color="text.secondary">
                  Detected cameras
                </Typography>
                <Button
                  size="small"
                  startIcon={camDetecting ? <CircularProgress size={12} /> : <RefreshIcon fontSize="small" />}
                  onClick={runDetect}
                  disabled={camDetecting}
                  sx={{ py: 0, minHeight: 0, fontSize: 11 }}
                >
                  {camDetecting ? "Detecting…" : "Re-scan"}
                </Button>
              </Box>

              {detectedCams.length > 0 ? (
                <TextField
                  select
                  label="Camera"
                  value={form.webcamIndex}
                  onChange={(e) => setForm((f) => ({ ...f, webcamIndex: Number(e.target.value) }))}
                  fullWidth
                  helperText={
                    `${detectedCams.length} camera${detectedCams.length !== 1 ? "s" : ""} detected. ` +
                    "OpenCV index must match the PC running the vision worker (see scripts/list_opencv_cameras.py). " +
                    "Device / group IDs are from the browser, not TCP ports."
                  }
                >
                  {detectedCams.map((cam) => (
                    <MenuItem key={`${cam.deviceId}-${cam.index}`} value={cam.index}>
                      <Box display="flex" alignItems="flex-start" gap={1} py={0.5}>
                        <UsbIcon fontSize="small" sx={{ opacity: 0.6, mt: 0.25 }} />
                        <Box flex={1} minWidth={0}>
                          <Box display="flex" alignItems="center" gap={0.75} flexWrap="wrap">
                            <Typography variant="body2">{cam.label}</Typography>
                            {isLikelyBuiltInCamera(cam.label) ? (
                              <Chip size="small" label="Built-in / HD" sx={{ height: 20, fontSize: 10 }} />
                            ) : null}
                          </Box>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            component="div"
                            title={cam.deviceId}
                            sx={{ fontFamily: "monospace", fontSize: 10, wordBreak: "break-all" }}
                          >
                            deviceId {formatOpaqueId(cam.deviceId)}
                          </Typography>
                          {cam.groupId ? (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              component="div"
                              title={cam.groupId}
                              sx={{ fontFamily: "monospace", fontSize: 10, wordBreak: "break-all" }}
                            >
                              groupId {formatOpaqueId(cam.groupId)}
                            </Typography>
                          ) : null}
                          <Typography variant="caption" color="text.secondary" component="div">
                            OpenCV index {cam.index} (saved as webcam:{cam.index})
                          </Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </TextField>
              ) : camDetecting ? (
                <Box display="flex" alignItems="center" gap={1} py={1}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Requesting camera access…
                  </Typography>
                </Box>
              ) : (
                <Alert severity="warning" sx={{ py: 0.5 }}>
                  No cameras detected. Click <strong>Re-scan</strong> or allow camera permission in
                  your browser.
                </Alert>
              )}
            </Box>
          )}

          <TextField
            select
            label="Gate"
            value={form.gate}
            onChange={(e) => setForm((f) => ({ ...f, gate: e.target.value as GateType }))}
            SelectProps={{ native: true }}
            fullWidth
          >
            {GATE_OPTIONS.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </TextField>

          {dialog !== "new" && (
            <FormControlLabel
              control={
                <Switch
                  checked={form.active}
                  onChange={(_, v) => setForm((f) => ({ ...f, active: v }))}
                />
              }
              label="Vision worker ingests this camera when running"
            />
          )}

          {form.sourceType === "webcam" && (
            <Alert severity="info" icon={<UsbIcon />}>
              The vision worker must run <strong>natively</strong> on the same machine as the camera
              (outside Docker on Mac/Windows). The worker opens the HD stream via DirectShow
              (Windows) or AVFoundation (Mac) and requests 1280×720. To see which index OpenCV uses on
              that PC, run{" "}
              <code style={{ wordBreak: "break-all" }}>python scripts/list_opencv_cameras.py</code>.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button
            variant="contained"
            disabled={saveMut.isPending || !canSave}
            onClick={() => saveMut.mutate()}
          >
            {saveMut.isPending ? <CircularProgress size={18} /> : dialog === "new" ? "Add" : "Save"}
          </Button>
        </DialogActions>
      </Dialog>

      {probeMut.isSuccess && probeMut.data?.tcp_reachable && (
        <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>
          OK — status: {probeMut.data.status}.
        </Typography>
      )}
    </Box>
  );
}
