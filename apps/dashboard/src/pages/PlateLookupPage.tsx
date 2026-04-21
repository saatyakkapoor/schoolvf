import SearchIcon from "@mui/icons-material/Search";
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import TimelineIcon from "@mui/icons-material/Timeline";
import EventIcon from "@mui/icons-material/Event";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  InputAdornment,
  LinearProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { format, parseISO } from "date-fns";

import { getPlateDetail } from "../api/plates";
import { getApiErrorMessage } from "../utils/errors";

const GOLD = "#FFD700";

function formatDateTime(iso: string | null): string {
  if (!iso) return "\u2014";
  try {
    return format(parseISO(iso), "dd MMM yyyy, hh:mm:ss a");
  } catch {
    return iso;
  }
}

function statusColor(status: string): "success" | "warning" | "default" {
  if (status === "inside") return "success";
  if (status === "outside") return "warning";
  return "default";
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "\u2014";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  const parts: string[] = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}

export default function PlateLookupPage() {
  const navigate = useNavigate();
  const [input, setInput] = useState("");
  const [plate, setPlate] = useState<string | null>(null);

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["plate", plate],
    queryFn: () => getPlateDetail(plate!),
    enabled: Boolean(plate),
  });

  const onSearch = () => {
    const p = input.trim().toUpperCase();
    if (p) setPlate(p);
  };

  const is404 = isError && (error as any)?.response?.status === 404;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <SearchIcon sx={{ fontSize: 32, color: GOLD }} />
        <Typography variant="h4" fontWeight={700} sx={{ color: "#fff" }}>
          Plate Lookup
        </Typography>
      </Box>

      {/* Search bar */}
      <Paper
        sx={{
          p: 3,
          mb: 4,
          bgcolor: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,215,0,0.12)",
          borderRadius: 3,
          display: "flex",
          gap: 2,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <TextField
          label="Enter plate number"
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === "Enter" && onSearch()}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon sx={{ color: GOLD }} />
              </InputAdornment>
            ),
            sx: {
              fontFamily: "monospace",
              fontWeight: 700,
              fontSize: 18,
              letterSpacing: 2,
            },
          }}
          sx={{ flex: 1, minWidth: 240 }}
        />
        <Button
          variant="contained"
          size="large"
          onClick={onSearch}
          disabled={!input.trim()}
          sx={{
            bgcolor: GOLD,
            color: "#000",
            fontWeight: 700,
            px: 4,
            "&:hover": { bgcolor: "#e6c200" },
          }}
        >
          Search
        </Button>
        {plate && (
          <Button
            variant="outlined"
            onClick={() => refetch()}
            disabled={isLoading}
            sx={{ borderColor: "rgba(255,215,0,0.3)", color: GOLD }}
          >
            Refresh
          </Button>
        )}
      </Paper>

      {/* Loading */}
      {plate && isLoading && (
        <Box display="flex" justifyContent="center" py={6}>
          <CircularProgress sx={{ color: GOLD }} />
        </Box>
      )}

      {/* Error (not 404) */}
      {plate && isError && !is404 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {getApiErrorMessage(error)}
        </Alert>
      )}

      {/* 404 - plate not found */}
      {is404 && (
        <Paper
          sx={{
            py: 6,
            px: 4,
            textAlign: "center",
            bgcolor: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,215,0,0.08)",
            borderRadius: 3,
          }}
        >
          <DirectionsBusIcon sx={{ fontSize: 64, color: "rgba(255,215,0,0.2)", mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No records for plate "{plate}"
          </Typography>
          <Typography variant="body2" color="text.disabled" maxWidth={480} mx="auto" mb={3}>
            This plate has not been detected by any camera. Check the plate number for typos or
            ensure the vehicle is registered in the fleet.
          </Typography>
          <Button
            variant="outlined"
            startIcon={<ArrowForwardIcon />}
            onClick={() => navigate("/vehicles")}
            sx={{ borderColor: GOLD, color: GOLD, "&:hover": { borderColor: GOLD, bgcolor: "rgba(255,215,0,0.08)" } }}
          >
            Register in Fleet
          </Button>
        </Paper>
      )}

      {/* Plate detail */}
      {plate && data && (
        <Box>
          {/* Info card */}
          <Card
            sx={{
              mb: 3,
              bgcolor: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,215,0,0.15)",
              borderRadius: 3,
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <Typography
                    variant="h3"
                    sx={{
                      fontFamily: "monospace",
                      fontWeight: 800,
                      letterSpacing: 3,
                      color: GOLD,
                      textAlign: { xs: "center", sm: "left" },
                    }}
                  >
                    {data.plate_number}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={8}>
                  <Grid container spacing={2}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.disabled" display="block">
                        STATUS
                      </Typography>
                      <Chip
                        label={data.current_status.toUpperCase()}
                        color={statusColor(data.current_status)}
                        size="small"
                        sx={{ fontWeight: 700, mt: 0.5 }}
                      />
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.disabled" display="block">
                        LAST SEEN
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        {formatDateTime(data.last_seen)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.disabled" display="block">
                        LAST CAMERA
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        {data.last_camera ?? "\u2014"}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.disabled" display="block">
                        TOTAL TRIPS
                      </Typography>
                      <Typography variant="h5" fontWeight={700} sx={{ color: GOLD, mt: 0.5 }}>
                        {data.total_trips}
                      </Typography>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Recent Trips */}
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
              <TimelineIcon sx={{ color: GOLD, fontSize: 20 }} />
              <Typography variant="subtitle1" fontWeight={700}>
                Recent Trips
              </Typography>
            </Box>
            {data.recent_trips.length === 0 ? (
              <Typography variant="body2" color="text.disabled" sx={{ pl: 3.5 }}>
                No trip records for this plate yet.
              </Typography>
            ) : (
              <Paper
                sx={{
                  bgcolor: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,215,0,0.08)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow
                        sx={{
                          "& th": {
                            fontWeight: 700,
                            color: GOLD,
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: 1,
                            borderBottom: "1px solid rgba(255,215,0,0.12)",
                          },
                        }}
                      >
                        <TableCell>Status</TableCell>
                        <TableCell>Exit</TableCell>
                        <TableCell>Entry</TableCell>
                        <TableCell>Duration</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.recent_trips.map((t) => (
                        <TableRow
                          key={t.id}
                          sx={{
                            "&:hover": { bgcolor: "rgba(255,215,0,0.04)" },
                            "& td": { borderBottom: "1px solid rgba(255,255,255,0.04)" },
                          }}
                        >
                          <TableCell>
                            <Chip
                              label={t.status.toUpperCase()}
                              color={t.status === "closed" ? "success" : t.status === "open" ? "warning" : "error"}
                              size="small"
                              sx={{ fontWeight: 700, fontSize: 11 }}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {formatDateTime(t.exit_time)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {formatDateTime(t.entry_time)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: "monospace" }}>
                              {formatDuration(t.duration_seconds)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            )}
          </Box>

          {/* Recent Events */}
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
              <EventIcon sx={{ color: GOLD, fontSize: 20 }} />
              <Typography variant="subtitle1" fontWeight={700}>
                Recent Events
              </Typography>
            </Box>
            {data.recent_events.length === 0 ? (
              <Typography variant="body2" color="text.disabled" sx={{ pl: 3.5 }}>
                No event records for this plate yet.
              </Typography>
            ) : (
              <Paper
                sx={{
                  bgcolor: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,215,0,0.08)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow
                        sx={{
                          "& th": {
                            fontWeight: 700,
                            color: GOLD,
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: 1,
                            borderBottom: "1px solid rgba(255,215,0,0.12)",
                          },
                        }}
                      >
                        <TableCell>Time</TableCell>
                        <TableCell>Gate</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Review</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.recent_events.map((e) => {
                        const pct = Math.round(e.confidence * 100);
                        const confColor =
                          pct >= 80 ? "success.main" : pct >= 60 ? "warning.main" : "error.main";
                        return (
                          <TableRow
                            key={e.id}
                            sx={{
                              "&:hover": { bgcolor: "rgba(255,215,0,0.04)" },
                              "& td": { borderBottom: "1px solid rgba(255,255,255,0.04)" },
                            }}
                          >
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {formatDateTime(e.timestamp)}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={e.gate_type.toUpperCase()}
                                color={e.gate_type === "exit" ? "warning" : "success"}
                                size="small"
                                sx={{ fontWeight: 700, fontSize: 11 }}
                              />
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 100 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={pct}
                                  sx={{
                                    flex: 1,
                                    height: 5,
                                    borderRadius: 3,
                                    bgcolor: "rgba(255,255,255,0.08)",
                                    "& .MuiLinearProgress-bar": { bgcolor: confColor, borderRadius: 3 },
                                  }}
                                />
                                <Typography variant="caption" sx={{ fontFamily: "monospace", fontWeight: 600 }}>
                                  {pct}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={e.review_status.toUpperCase()}
                                variant="outlined"
                                size="small"
                                sx={{ fontSize: 11 }}
                              />
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            )}
          </Box>

          {/* Navigate to fleet */}
          <Button
            variant="outlined"
            startIcon={<ArrowForwardIcon />}
            onClick={() => navigate("/vehicles")}
            sx={{
              borderColor: "rgba(255,215,0,0.3)",
              color: GOLD,
              "&:hover": { borderColor: GOLD, bgcolor: "rgba(255,215,0,0.08)" },
            }}
          >
            View in Fleet
          </Button>
        </Box>
      )}
    </Box>
  );
}
