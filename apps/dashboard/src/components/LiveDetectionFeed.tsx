/**
 * LiveDetectionFeed — real-time plate detection list.
 *
 * Mismatch detection: when the vision worker reads a route placard/LED display
 * that conflicts with the plate's registered route, the row is highlighted in
 * amber with a warning badge and an inline "Adjust" form for staff to classify
 * the event (temporary swap, permanent change, glitch, other).
 */
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import FiberManualRecordIcon from "@mui/icons-material/FiberManualRecord";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  FormControl,
  FormControlLabel,
  List,
  ListItem,
  Radio,
  RadioGroup,
  Skeleton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { adjustDetection, getLiveRecent } from "../api/live";
import { useLiveWebSocket } from "../hooks/useLiveWebSocket";
import type { LiveDetection } from "../types";

interface LiveDetectionFeedProps {
  maxItems?: number;
  height?: number;
  cameraId?: string;
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return iso;
  }
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "success" : pct >= 60 ? "warning" : "error";
  return (
    <Chip
      label={`${pct}%`}
      size="small"
      color={color}
      variant="outlined"
      sx={{ fontFamily: "monospace", fontSize: 11, height: 20, minWidth: 44 }}
    />
  );
}

function RouteBadge({ routeNumber, routeName, driverName }: { routeNumber: string; routeName: string; driverName?: string }) {
  const label = routeName ? `${routeNumber} · ${routeName}` : routeNumber;
  return (
    <Tooltip title={driverName ? `Driver: ${driverName}` : label} placement="top" arrow>
      <Chip
        icon={<DirectionsBusIcon sx={{ fontSize: "12px !important" }} />}
        label={label}
        size="small"
        sx={{
          height: 20, fontSize: 11, fontWeight: 700,
          bgcolor: "rgba(255,215,0,0.12)", color: "#FFD700",
          border: "1px solid rgba(255,215,0,0.35)",
          "& .MuiChip-icon": { color: "#FFD700" },
        }}
      />
    </Tooltip>
  );
}

const SWAP_REASONS = [
  { value: "temporary", label: "Temporary bus swap (back soon)" },
  { value: "permanent", label: "Permanent route reassignment" },
  { value: "glitch",    label: "OCR / system glitch — ignore" },
  { value: "other",     label: "Other" },
];

function AdjustForm({ row, onDone }: { row: LiveDetection; onDone: () => void }) {
  const [reason, setReason] = useState("temporary");
  const [notes, setNotes] = useState("");

  const mut = useMutation({
    mutationFn: () => adjustDetection(row.id, { swap_type: reason, notes: notes.trim() || null }),
    onSuccess: onDone,
  });

  return (
    <Box
      sx={{
        mt: 1, p: 1.5, borderRadius: 1,
        bgcolor: "rgba(255,152,0,0.06)",
        border: "1px solid rgba(255,152,0,0.25)",
      }}
    >
      <Typography variant="caption" fontWeight={700} color="warning.main" sx={{ mb: 1, display: "block" }}>
        What happened?
      </Typography>
      <FormControl size="small" fullWidth>
        <RadioGroup value={reason} onChange={(_, v) => setReason(v)}>
          {SWAP_REASONS.map((r) => (
            <FormControlLabel
              key={r.value}
              value={r.value}
              control={<Radio size="small" sx={{ py: 0.3 }} />}
              label={<Typography variant="caption">{r.label}</Typography>}
            />
          ))}
        </RadioGroup>
      </FormControl>
      <TextField
        placeholder="Notes (optional)"
        size="small"
        fullWidth
        multiline
        minRows={1}
        maxRows={3}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        sx={{ mt: 1, "& .MuiInputBase-input": { fontSize: 12 } }}
      />
      <Stack direction="row" spacing={1} mt={1} justifyContent="flex-end">
        <Button size="small" onClick={onDone} disabled={mut.isPending}>Cancel</Button>
        <Button
          size="small"
          variant="contained"
          color="warning"
          onClick={() => mut.mutate()}
          disabled={mut.isPending}
          startIcon={mut.isPending ? <CircularProgress size={12} /> : undefined}
        >
          Save
        </Button>
      </Stack>
      {mut.isError && (
        <Alert severity="error" sx={{ mt: 1, py: 0, fontSize: 11 }}>
          Failed to save. Try again.
        </Alert>
      )}
    </Box>
  );
}

function DetectionRow({ row }: { row: LiveDetection }) {
  const hasRoute = row.is_registered && row.route_number;
  const isMismatch = row.is_mismatch && !row.swap_resolved;
  const isResolved = row.is_mismatch && row.swap_resolved;
  const [adjustOpen, setAdjustOpen] = useState(false);

  const borderColor = isMismatch
    ? "rgba(255,152,0,0.6)"
    : isResolved
    ? "rgba(76,175,80,0.4)"
    : "transparent";

  return (
    <ListItem
      disableGutters
      sx={{
        px: 1.5, py: 1, gap: 1.5,
        alignItems: "flex-start",
        flexDirection: "column",
        "&:hover": { bgcolor: "action.hover" },
        borderRadius: 1,
        borderLeft: `3px solid ${borderColor}`,
        bgcolor: isMismatch ? "rgba(255,152,0,0.04)" : isResolved ? "rgba(76,175,80,0.03)" : "transparent",
        transition: "background-color 0.2s",
      }}
    >
      {/* Main row */}
      <Box display="flex" gap={1.5} alignItems="flex-start" width="100%">
        {/* Snapshot */}
        <Box
          sx={{
            flexShrink: 0, width: 88, height: 50, borderRadius: 1,
            overflow: "hidden", bgcolor: "action.disabledBackground",
            display: "flex", alignItems: "center", justifyContent: "center",
            border: isMismatch ? "1px solid rgba(255,152,0,0.5)" : hasRoute ? "1px solid rgba(255,215,0,0.3)" : "none",
          }}
        >
          {row.snapshot_base64 ? (
            <Box
              component="img"
              src={`data:image/jpeg;base64,${row.snapshot_base64}`}
              alt={row.plate_text}
              sx={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          ) : (
            <Typography variant="caption" color="text.disabled" sx={{ fontSize: 10 }}>No img</Typography>
          )}
        </Box>

        {/* Plate + meta */}
        <Box flex={1} minWidth={0}>
          <Stack direction="row" alignItems="center" spacing={0.75} flexWrap="wrap" useFlexGap>
            <Typography
              variant="body2" fontWeight={700} fontFamily="monospace"
              sx={{ letterSpacing: 1.5, color: isMismatch ? "warning.main" : hasRoute ? "#FFD700" : "text.primary" }}
            >
              {row.plate_text}
            </Typography>
            <ConfidenceBadge value={row.confidence} />

            {/* Registered route from DB */}
            {hasRoute && (
              <RouteBadge routeNumber={row.route_number!} routeName={row.route_name || ""} driverName={row.driver_name} />
            )}

            {/* Detected route from bus placard / LED display — always show when present */}
            {row.detected_route && (
              <Tooltip
                title={
                  isResolved
                    ? `Resolved as: ${row.swap_type}`
                    : row.is_mismatch
                    ? row.route_number
                      ? `Plate registered to ${row.route_number} but bus shows ${row.detected_route}`
                      : `Unregistered plate showing route ${row.detected_route}`
                    : `Route read from bus placard / display`
                }
                arrow
              >
                <Chip
                  icon={
                    isResolved
                      ? <CheckCircleIcon sx={{ fontSize: "12px !important" }} />
                      : row.is_mismatch
                      ? <WarningAmberIcon sx={{ fontSize: "12px !important" }} />
                      : <DirectionsBusIcon sx={{ fontSize: "12px !important" }} />
                  }
                  label={
                    isResolved
                      ? `${row.detected_route} · ${row.swap_type}`
                      : row.is_mismatch
                      ? `⚠ ${row.detected_route}`
                      : row.detected_route
                  }
                  size="small"
                  color={isResolved ? "success" : row.is_mismatch ? "warning" : "info"}
                  variant={row.is_mismatch ? "filled" : "outlined"}
                  sx={{ height: 20, fontSize: 11, fontWeight: 700 }}
                />
              </Tooltip>
            )}
          </Stack>

          <Stack direction="row" alignItems="center" spacing={0.5} mt={0.25}>
            <Typography variant="caption" color="text.secondary" noWrap>
              {row.camera_name} · <span style={{ fontFamily: "monospace", fontSize: 10 }}>{row.camera_id}</span> · {formatTime(row.detected_at)}
            </Typography>
          </Stack>

          {/* Mismatch action buttons */}
          {isMismatch && !adjustOpen && (
            <Stack direction="row" spacing={0.75} mt={0.75}>
              <Button
                size="small"
                variant="outlined"
                color="warning"
                startIcon={<SwapHorizIcon />}
                onClick={() => setAdjustOpen(true)}
                sx={{ fontSize: 11, py: 0.25, px: 1 }}
              >
                Adjust
              </Button>
            </Stack>
          )}

          {/* Resolved info */}
          {isResolved && (
            <Stack direction="row" alignItems="center" spacing={0.5} mt={0.5}>
              <CheckCircleIcon sx={{ fontSize: 13, color: "success.main" }} />
              <Typography variant="caption" color="success.main">
                {row.swap_type} — by {row.swap_resolved_by}
                {row.swap_notes ? ` · ${row.swap_notes}` : ""}
              </Typography>
            </Stack>
          )}
        </Box>
      </Box>

      {/* Inline adjust form */}
      <Collapse in={adjustOpen} unmountOnExit sx={{ width: "100%" }}>
        <AdjustForm row={row} onDone={() => setAdjustOpen(false)} />
      </Collapse>
    </ListItem>
  );
}

export default function LiveDetectionFeed({ maxItems = 60, height = 400, cameraId }: LiveDetectionFeedProps) {
  const { connected, recent, snapshotReceived } = useLiveWebSocket();

  const { data: restData, isLoading } = useQuery({
    queryKey: ["live", "recent", 100],
    queryFn: () => getLiveRecent(100),
    staleTime: 5_000,
    enabled: !snapshotReceived,
  });

  const allRows = snapshotReceived ? recent : (restData ?? []);
  const rows = (cameraId ? allRows.filter((r) => r.camera_id === cameraId) : allRows).slice(0, maxItems);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height, border: "1px solid", borderColor: "divider", borderRadius: 2, overflow: "hidden" }}>
      {/* Header */}
      <Stack
        direction="row" alignItems="center" spacing={1}
        sx={{ px: 1.5, py: 1, bgcolor: "background.paper", borderBottom: "1px solid", borderColor: "divider" }}
      >
        <FiberManualRecordIcon
          sx={{
            fontSize: 10,
            color: connected ? "success.main" : "warning.main",
            "@keyframes pulse": { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0.3 } },
            animation: connected ? "pulse 1.4s ease-in-out infinite" : "none",
          }}
        />
        <Typography variant="subtitle2" sx={{ flex: 1 }}>Plate detections</Typography>
        <Typography variant="caption" color="text.secondary">{connected ? "Live" : "Connecting…"}</Typography>
      </Stack>

      {/* List */}
      <Box sx={{ overflowY: "auto", flex: 1 }}>
        {isLoading && rows.length === 0 ? (
          <Box sx={{ p: 1.5 }}>
            {[0, 1, 2].map((i) => <Skeleton key={i} variant="rectangular" height={62} sx={{ mb: 1, borderRadius: 1 }} />)}
          </Box>
        ) : rows.length === 0 ? (
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", p: 2, gap: 1 }}>
            <Typography variant="body2" color="text.secondary" textAlign="center">
              {cameraId ? "No detections for this camera yet." : "No detections yet."}
            </Typography>
            <Typography variant="caption" color="text.disabled" textAlign="center">
              Entry and exit both appear here. Tabs only change the video stream.
            </Typography>
          </Box>
        ) : (
          <List disablePadding dense>
            {rows.map((row, idx) => (
              <Box key={row.id}>
                <DetectionRow row={row} />
                {idx < rows.length - 1 && <Divider sx={{ mx: 1.5 }} />}
              </Box>
            ))}
          </List>
        )}
      </Box>
    </Box>
  );
}
