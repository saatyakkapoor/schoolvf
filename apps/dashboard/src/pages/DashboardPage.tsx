/**
 * DashboardPage — overview with live summary stats + recent detection ticker.
 *
 * - Summary cards pulled from /api/dashboard/summary (auto-refreshed every 30s).
 * - Live detection ticker (last 5) via WebSocket — updates in real time.
 * - No mock data. Cards show 0 until cameras + vision worker produce real events.
 */
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import ScheduleIcon from "@mui/icons-material/Schedule";
import SpeedIcon from "@mui/icons-material/Speed";
import TimelineIcon from "@mui/icons-material/Timeline";
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  Skeleton,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";

import { getDashboardSummary } from "../api/dashboard";
import LiveDetectionFeed from "../components/LiveDetectionFeed";
import { useLiveWebSocket } from "../hooks/useLiveWebSocket";
import { getApiErrorMessage } from "../utils/errors";

// ── Summary stat card ────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  value: number | string;
  icon: ReactNode;
  accent?: "default" | "warning" | "error" | "success";
}

function StatCard({ label, value, icon, accent = "default" }: StatCardProps) {
  const accentColor = {
    default: "primary.main",
    warning: "warning.main",
    error: "error.main",
    success: "success.main",
  }[accent];

  return (
    <Card variant="outlined" sx={{ height: "100%" }}>
      <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
        <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography variant="caption" color="text.secondary">
              {label}
            </Typography>
            <Typography variant="h4" fontWeight={600} sx={{ mt: 0.25 }}>
              {value}
            </Typography>
          </Box>
          <Box
            sx={{
              p: 1,
              borderRadius: 1.5,
              bgcolor: `${accentColor}18`,
              color: accentColor,
              display: "flex",
              alignItems: "center",
            }}
          >
            {icon}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { connected, recent } = useLiveWebSocket();

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["dashboard", "summary"],
    queryFn: getDashboardSummary,
    refetchInterval: 30_000, // auto-refresh every 30s
    staleTime: 20_000,
  });

  // Last 5 detections for the ticker — comes from WebSocket (no polling)
  const tickerRows = recent.slice(0, 5);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
      {/* Page header */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 1 }}>
        <Typography variant="h5" fontWeight={600}>
          Overview
        </Typography>
        <Chip
          size="small"
          label={connected ? "Live" : "Connecting…"}
          color={connected ? "success" : "default"}
          variant="outlined"
          sx={{ fontSize: 11 }}
        />
      </Box>

      {/* Error banner (doesn't block ticker) */}
      {isError && (
        <Alert severity="warning" icon={<ErrorOutlineIcon />}>
          {getApiErrorMessage(error)} — summary values may be stale.
        </Alert>
      )}

      {/* ── Summary cards ─────────────────────────────────────────── */}
      <Grid container spacing={2}>
        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Buses tracked"
              value={data?.total_buses_known ?? 0}
              icon={<DirectionsBusIcon />}
            />
          )}
        </Grid>

        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Outside now"
              value={data?.buses_outside_now ?? 0}
              icon={<SpeedIcon />}
              accent={data?.buses_outside_now ? "warning" : "default"}
            />
          )}
        </Grid>

        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Overdue"
              value={data?.buses_overdue ?? 0}
              icon={<ScheduleIcon />}
              accent={data?.buses_overdue ? "error" : "default"}
            />
          )}
        </Grid>

        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Alerts today"
              value={data?.alerts_today ?? 0}
              icon={<NotificationsActiveIcon />}
              accent={data?.alerts_today ? "error" : "default"}
            />
          )}
        </Grid>

        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Events today"
              value={data?.events_today ?? 0}
              icon={<TimelineIcon />}
              accent="success"
            />
          )}
        </Grid>

        <Grid item xs={6} sm={4} md={2}>
          {isLoading ? (
            <Skeleton variant="rectangular" height={90} sx={{ borderRadius: 1 }} />
          ) : (
            <StatCard
              label="Trips today"
              value={data?.trips_today ?? 0}
              icon={<DirectionsBusIcon />}
            />
          )}
        </Grid>
      </Grid>

      {/* ── Live section ──────────────────────────────────────────── */}
      <Grid container spacing={2}>
        {/* Left: live detection feed (all cameras) */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Live plate detections
          </Typography>
          <LiveDetectionFeed height={380} maxItems={40} />
        </Grid>

        {/* Right: last 5 plate ticker + active alerts */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Recent activity
          </Typography>
          <Card variant="outlined" sx={{ height: 380, display: "flex", flexDirection: "column" }}>
            <CardContent sx={{ p: 0, flex: 1, overflow: "auto", "&:last-child": { pb: 0 } }}>
              {tickerRows.length === 0 ? (
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    gap: 1,
                    p: 3,
                  }}
                >
                  <Typography variant="body2" color="text.secondary" textAlign="center">
                    Waiting for detections…
                  </Typography>
                  <Typography variant="caption" color="text.disabled" textAlign="center">
                    Start the vision worker to see plates appear here in real time.
                  </Typography>
                </Box>
              ) : (
                tickerRows.map((row, idx) => (
                  <Box key={row.id}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, px: 2, py: 1.25 }}>
                      {row.snapshot_base64 ? (
                        <Box
                          component="img"
                          src={`data:image/jpeg;base64,${row.snapshot_base64}`}
                          alt={row.plate_text}
                          sx={{ width: 72, height: 40, objectFit: "cover", borderRadius: 1, flexShrink: 0 }}
                        />
                      ) : (
                        <Box
                          sx={{
                            width: 72,
                            height: 40,
                            bgcolor: "action.disabledBackground",
                            borderRadius: 1,
                            flexShrink: 0,
                          }}
                        />
                      )}
                      <Box flex={1} minWidth={0}>
                        <Stack direction="row" alignItems="center" spacing={0.75} flexWrap="wrap" useFlexGap>
                          <Typography
                            variant="body2"
                            fontWeight={700}
                            fontFamily="monospace"
                            sx={{ letterSpacing: 1, color: row.is_registered ? "#FFD700" : "text.primary" }}
                          >
                            {row.plate_text}
                          </Typography>
                          {row.is_registered && row.route_number && (
                            <Tooltip title={row.route_name || row.route_number} arrow>
                              <Chip
                                icon={<DirectionsBusIcon sx={{ fontSize: "11px !important" }} />}
                                label={row.route_number}
                                size="small"
                                sx={{
                                  height: 18,
                                  fontSize: 10,
                                  fontWeight: 700,
                                  bgcolor: "rgba(255,215,0,0.12)",
                                  color: "#FFD700",
                                  border: "1px solid rgba(255,215,0,0.35)",
                                  "& .MuiChip-icon": { color: "#FFD700" },
                                }}
                              />
                            </Tooltip>
                          )}
                        </Stack>
                        <Typography variant="caption" color="text.secondary" noWrap>
                          {row.camera_name} · {Math.round(row.confidence * 100)}% conf
                        </Typography>
                      </Box>
                      <Typography variant="caption" color="text.disabled" sx={{ flexShrink: 0 }}>
                        {new Date(row.detected_at).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </Typography>
                    </Box>
                    {idx < tickerRows.length - 1 && <Divider />}
                  </Box>
                ))
              )}

              {/* Active alerts from API */}
              {data?.active_alerts && data.active_alerts.length > 0 && (
                <>
                  <Divider sx={{ my: 1 }} />
                  <Box sx={{ px: 2, pb: 1 }}>
                    <Typography variant="caption" color="text.secondary" fontWeight={600}>
                      ACTIVE ALERTS
                    </Typography>
                    {data.active_alerts.slice(0, 5).map((alert) => (
                      <Box key={alert.id} sx={{ mt: 0.75 }}>
                        <Typography variant="body2" color="warning.main" fontWeight={500}>
                          {alert.plate_number} — {alert.alert_type}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {alert.message}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
