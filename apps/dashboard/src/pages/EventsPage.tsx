import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import EventIcon from "@mui/icons-material/Event";
import SensorsIcon from "@mui/icons-material/Sensors";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { format, parseISO } from "date-fns";

import { getEvents } from "../api/events";
import { getApiErrorMessage } from "../utils/errors";
import type { GateType, ReviewStatus } from "../types";

const GOLD = "#FFD700";

function formatDateTime(iso: string): string {
  try {
    return format(parseISO(iso), "dd MMM yyyy, hh:mm:ss a");
  } catch {
    return iso;
  }
}

function reviewColor(status: string): "default" | "success" | "warning" | "error" | "info" {
  switch (status) {
    case "approved":
      return "success";
    case "corrected":
      return "info";
    case "rejected":
      return "error";
    default:
      return "warning";
  }
}

export default function EventsPage() {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [plateFilter, setPlateFilter] = useState("");
  const [gateFilter, setGateFilter] = useState<string>("all");
  const [reviewFilter, setReviewFilter] = useState<string>("all");

  const filters: Record<string, unknown> = {
    page: page + 1,
    page_size: pageSize,
  };
  if (plateFilter.trim()) filters.plate_number = plateFilter.trim().toUpperCase();
  if (gateFilter !== "all") filters.gate_type = gateFilter as GateType;
  if (reviewFilter !== "all") filters.review_status = reviewFilter as ReviewStatus;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["events", filters],
    queryFn: () => getEvents(filters as any),
    placeholderData: (prev) => prev,
  });

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <EventIcon sx={{ fontSize: 32, color: GOLD }} />
        <Typography variant="h4" fontWeight={700} sx={{ color: "#fff" }}>
          Gate Events
        </Typography>
      </Box>

      {/* Filter bar */}
      <Paper
        sx={{
          p: 2,
          mb: 3,
          bgcolor: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,215,0,0.12)",
          borderRadius: 2,
          display: "flex",
          gap: 2,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <TextField
          label="Plate number"
          size="small"
          value={plateFilter}
          onChange={(e) => {
            setPlateFilter(e.target.value.toUpperCase());
            setPage(0);
          }}
          sx={{ minWidth: 160 }}
          InputProps={{ sx: { fontFamily: "monospace", fontWeight: 700 } }}
        />
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Gate type</InputLabel>
          <Select
            value={gateFilter}
            label="Gate type"
            onChange={(e) => {
              setGateFilter(e.target.value);
              setPage(0);
            }}
          >
            <MenuItem value="all">All gates</MenuItem>
            <MenuItem value="entry">Entry</MenuItem>
            <MenuItem value="exit">Exit</MenuItem>
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Review status</InputLabel>
          <Select
            value={reviewFilter}
            label="Review status"
            onChange={(e) => {
              setReviewFilter(e.target.value);
              setPage(0);
            }}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="approved">Approved</MenuItem>
            <MenuItem value="corrected">Corrected</MenuItem>
            <MenuItem value="rejected">Rejected</MenuItem>
          </Select>
        </FormControl>
      </Paper>

      {/* Error */}
      {isError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {getApiErrorMessage(error)}
        </Alert>
      )}

      {/* Loading */}
      {isLoading && !data && (
        <Box display="flex" justifyContent="center" py={8}>
          <CircularProgress sx={{ color: GOLD }} />
        </Box>
      )}

      {/* Empty state */}
      {data && data.items.length === 0 ? (
        <Paper
          sx={{
            py: 8,
            px: 4,
            textAlign: "center",
            bgcolor: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,215,0,0.08)",
            borderRadius: 3,
          }}
        >
          <SensorsIcon sx={{ fontSize: 64, color: "rgba(255,215,0,0.25)", mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No gate events recorded
          </Typography>
          <Typography variant="body2" color="text.disabled" maxWidth={480} mx="auto">
            Gate events are created when the vision pipeline detects a license plate at an entry or
            exit camera. Start the vision worker to begin recording events.
          </Typography>
        </Paper>
      ) : data ? (
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
                      borderBottom: `1px solid rgba(255,215,0,0.15)`,
                      fontSize: 12,
                      textTransform: "uppercase",
                      letterSpacing: 1,
                    },
                  }}
                >
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Plate</TableCell>
                  <TableCell>Gate</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Review</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.items.map((e) => {
                  const pct = Math.round(e.confidence * 100);
                  const confColor =
                    pct >= 80 ? "success.main" : pct >= 60 ? "warning.main" : "error.main";
                  return (
                    <TableRow
                      key={e.id}
                      sx={{
                        "&:hover": { bgcolor: "rgba(255,215,0,0.04)" },
                        "& td": { borderBottom: "1px solid rgba(255,255,255,0.05)" },
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {formatDateTime(e.timestamp)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Typography
                            variant="body2"
                            sx={{ fontFamily: "monospace", fontWeight: 700, letterSpacing: 1, color: "#fff" }}
                          >
                            {e.plate_number}
                          </Typography>
                          {e.route_number && (
                            <Chip
                              icon={<DirectionsBusIcon sx={{ fontSize: "11px !important" }} />}
                              label={e.route_number}
                              size="small"
                              sx={{
                                height: 18,
                                fontSize: 10,
                                fontWeight: 700,
                                bgcolor: "rgba(255,215,0,0.12)",
                                color: GOLD,
                                border: "1px solid rgba(255,215,0,0.35)",
                                "& .MuiChip-icon": { color: GOLD },
                              }}
                            />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={e.gate_type.toUpperCase()}
                          color={e.gate_type === "exit" ? "warning" : "success"}
                          size="small"
                          sx={{ fontWeight: 700, fontSize: 11, minWidth: 64 }}
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 120 }}>
                          <LinearProgress
                            variant="determinate"
                            value={pct}
                            sx={{
                              flex: 1,
                              height: 6,
                              borderRadius: 3,
                              bgcolor: "rgba(255,255,255,0.08)",
                              "& .MuiLinearProgress-bar": {
                                bgcolor: confColor,
                                borderRadius: 3,
                              },
                            }}
                          />
                          <Typography
                            variant="caption"
                            sx={{ fontFamily: "monospace", fontWeight: 600, minWidth: 36 }}
                          >
                            {pct}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={e.review_status.toUpperCase()}
                          color={reviewColor(e.review_status)}
                          variant="outlined"
                          size="small"
                          sx={{ fontWeight: 600, fontSize: 11 }}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            component="div"
            count={data.total}
            page={page}
            onPageChange={(_, p) => setPage(p)}
            rowsPerPage={pageSize}
            onRowsPerPageChange={(e) => {
              setPageSize(parseInt(e.target.value, 10));
              setPage(0);
            }}
            rowsPerPageOptions={[10, 25, 50, 100]}
            sx={{
              borderTop: "1px solid rgba(255,215,0,0.08)",
              ".MuiTablePagination-toolbar": { minHeight: 48 },
            }}
          />
        </Paper>
      ) : null}
    </Box>
  );
}
