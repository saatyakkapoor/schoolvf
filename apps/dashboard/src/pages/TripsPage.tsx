import TimelineIcon from "@mui/icons-material/Timeline";
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
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

import { getTrips } from "../api/trips";
import { getApiErrorMessage } from "../utils/errors";
import type { TripStatus } from "../types";

const GOLD = "#FFD700";

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

function formatDateTime(iso: string | null): string {
  if (!iso) return "\u2014";
  try {
    return format(parseISO(iso), "dd MMM yyyy, hh:mm:ss a");
  } catch {
    return iso;
  }
}

function statusColor(status: string): "warning" | "success" | "error" {
  if (status === "open") return "warning";
  if (status === "closed") return "success";
  return "error";
}

function anomalyLabel(code: string): string {
  return code.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function TripsPage() {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [plateFilter, setPlateFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  const filters: Record<string, unknown> = {
    page: page + 1,
    page_size: pageSize,
  };
  if (plateFilter.trim()) filters.plate_number = plateFilter.trim().toUpperCase();
  if (statusFilter !== "all") filters.status = statusFilter as TripStatus;
  if (fromDate) filters.from_date = fromDate;
  if (toDate) filters.to_date = toDate;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["trips", filters],
    queryFn: () => getTrips(filters as any),
    placeholderData: (prev) => prev,
  });

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <TimelineIcon sx={{ fontSize: 32, color: GOLD }} />
        <Typography variant="h4" fontWeight={700} sx={{ color: "#fff" }}>
          Trip History
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
          <InputLabel>Status</InputLabel>
          <Select
            value={statusFilter}
            label="Status"
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(0);
            }}
          >
            <MenuItem value="all">All statuses</MenuItem>
            <MenuItem value="open">Open</MenuItem>
            <MenuItem value="closed">Closed</MenuItem>
            <MenuItem value="overdue">Overdue</MenuItem>
          </Select>
        </FormControl>
        <TextField
          label="From date"
          type="date"
          size="small"
          value={fromDate}
          onChange={(e) => {
            setFromDate(e.target.value);
            setPage(0);
          }}
          InputLabelProps={{ shrink: true }}
          sx={{ minWidth: 150 }}
        />
        <TextField
          label="To date"
          type="date"
          size="small"
          value={toDate}
          onChange={(e) => {
            setToDate(e.target.value);
            setPage(0);
          }}
          InputLabelProps={{ shrink: true }}
          sx={{ minWidth: 150 }}
        />
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

      {/* Table */}
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
          <DirectionsBusIcon sx={{ fontSize: 64, color: "rgba(255,215,0,0.25)", mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No trips recorded yet
          </Typography>
          <Typography variant="body2" color="text.disabled" maxWidth={480} mx="auto">
            Trips are created automatically when buses exit and re-enter the gate. Once the vision
            pipeline detects plate movements, trip records will appear here.
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
                  <TableCell>Plate</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Exit Time</TableCell>
                  <TableCell>Entry Time</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Anomaly</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.items.map((t) => (
                  <TableRow
                    key={t.id}
                    sx={{
                      "&:hover": { bgcolor: "rgba(255,215,0,0.04)" },
                      "& td": { borderBottom: "1px solid rgba(255,255,255,0.05)" },
                    }}
                  >
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{ fontFamily: "monospace", fontWeight: 700, letterSpacing: 1, color: "#fff" }}
                      >
                        {t.plate_number}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={t.status.toUpperCase()}
                        color={statusColor(t.status)}
                        size="small"
                        sx={{ fontWeight: 700, fontSize: 11, minWidth: 72 }}
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
                      <Typography variant="body2" sx={{ fontFamily: "monospace", color: "text.secondary" }}>
                        {formatDuration(t.duration_seconds)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {t.anomaly_code !== "none" ? (
                        <Chip
                          label={anomalyLabel(t.anomaly_code)}
                          color="error"
                          variant="outlined"
                          size="small"
                          sx={{ fontSize: 11 }}
                        />
                      ) : (
                        <Typography variant="caption" color="text.disabled">
                          None
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
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
