/**
 * ReportsPage — operator-facing reporting UI.
 *
 * Goal: pretty, quick filters + a "Download CSV" button so the school
 * can hand a clean spreadsheet to whoever needs daily / weekly attendance
 * style logs of bus gate events.
 *
 * The page uses the existing /events endpoint (paginated) but pulls all
 * pages on export so the CSV reflects the full filter window, not just
 * the visible 25 rows.
 */
import AssessmentIcon from "@mui/icons-material/Assessment";
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import DownloadIcon from "@mui/icons-material/Download";
import FilterAltIcon from "@mui/icons-material/FilterAlt";
import PrintIcon from "@mui/icons-material/Print";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  useTheme,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { format, parseISO, startOfDay, endOfDay, subDays } from "date-fns";
import { useMemo, useState } from "react";

import { getEvents } from "../api/events";
import type { EventFilters, GateEvent, GateType } from "../types";

const GOLD = "#FFD700";

type Range = "today" | "yesterday" | "last7" | "last30" | "custom";

function isoDay(d: Date): string {
  return d.toISOString();
}

function rangeBounds(range: Range, fromCustom: string, toCustom: string): { from: string; to: string } {
  const now = new Date();
  switch (range) {
    case "today":
      return { from: isoDay(startOfDay(now)), to: isoDay(endOfDay(now)) };
    case "yesterday": {
      const y = subDays(now, 1);
      return { from: isoDay(startOfDay(y)), to: isoDay(endOfDay(y)) };
    }
    case "last7":
      return { from: isoDay(startOfDay(subDays(now, 6))), to: isoDay(endOfDay(now)) };
    case "last30":
      return { from: isoDay(startOfDay(subDays(now, 29))), to: isoDay(endOfDay(now)) };
    case "custom":
      return {
        from: fromCustom ? isoDay(startOfDay(new Date(fromCustom))) : "",
        to: toCustom ? isoDay(endOfDay(new Date(toCustom))) : "",
      };
  }
}

function csvEscape(v: unknown): string {
  if (v == null) return "";
  const s = String(v);
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function buildCsv(rows: GateEvent[]): string {
  const header = [
    "timestamp",
    "plate_number",
    "route_number",
    "camera_name",
    "gate_type",
    "confidence",
    "review_status",
    "anomaly_code",
    "trip_id",
    "event_id",
  ];
  const lines = [header.join(",")];
  for (const r of rows) {
    lines.push(
      [
        r.timestamp,
        r.plate_number,
        r.route_number ?? "",
        r.camera_name,
        r.gate_type,
        Math.round((r.confidence || 0) * 100),
        r.review_status,
        r.anomaly_code,
        r.trip_id ?? "",
        r.id,
      ]
        .map(csvEscape)
        .join(","),
    );
  }
  return lines.join("\n");
}

async function fetchAllPages(filters: EventFilters): Promise<GateEvent[]> {
  const out: GateEvent[] = [];
  let page = 1;
  // 200 per page = fast, capped to 50 pages = 10k rows max for a single export
  for (let i = 0; i < 50; i++) {
    const resp = await getEvents({ ...filters, page, page_size: 200 });
    out.push(...resp.items);
    if (page >= resp.pages || resp.items.length === 0) break;
    page += 1;
  }
  return out;
}

function formatDateTime(iso: string): string {
  try {
    return format(parseISO(iso), "dd MMM yyyy, hh:mm:ss a");
  } catch {
    return iso;
  }
}

function StatCard({
  label,
  value,
  color = GOLD,
  icon,
}: {
  label: string;
  value: string | number;
  color?: string;
  icon?: React.ReactNode;
}) {
  return (
    <Paper
      sx={{
        p: 2,
        borderRadius: 2,
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: 0,
          width: 4,
          height: "100%",
          bgcolor: color,
        }}
      />
      <Stack direction="row" alignItems="center" spacing={1.5}>
        {icon && <Box sx={{ color, opacity: 0.85 }}>{icon}</Box>}
        <Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", textTransform: "uppercase", letterSpacing: 0.5, fontSize: 10 }}>
            {label}
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 700, color }}>
            {value}
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
}

export default function ReportsPage() {
  const theme = useTheme();
  const [range, setRange] = useState<Range>("today");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [plate, setPlate] = useState("");
  const [gate, setGate] = useState<string>("all");
  const [exporting, setExporting] = useState(false);

  const { from: fromIso, to: toIso } = useMemo(
    () => rangeBounds(range, from, to),
    [range, from, to],
  );

  const filters: EventFilters = useMemo(
    () => ({
      plate_number: plate.trim().toUpperCase() || undefined,
      gate_type: gate !== "all" ? (gate as GateType) : undefined,
      from_date: fromIso || undefined,
      to_date: toIso || undefined,
      page: 1,
      page_size: 50,
    }),
    [plate, gate, fromIso, toIso],
  );

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["reports", filters],
    queryFn: () => getEvents(filters),
    staleTime: 5_000,
  });

  const rows = data?.items ?? [];
  const total = data?.total ?? 0;

  const summary = useMemo(() => {
    const uniquePlates = new Set(rows.map((r) => r.plate_number)).size;
    const entries = rows.filter((r) => r.gate_type === "entry").length;
    const exits = rows.filter((r) => r.gate_type === "exit").length;
    const anomalies = rows.filter((r) => r.anomaly_code && r.anomaly_code !== "none").length;
    return { uniquePlates, entries, exits, anomalies };
  }, [rows]);

  const handleDownload = async () => {
    setExporting(true);
    try {
      const all = await fetchAllPages(filters);
      const csv = buildCsv(all);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const stamp = format(new Date(), "yyyyMMdd-HHmmss");
      a.href = url;
      a.download = `bus-events-${stamp}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const handleReset = () => {
    setRange("today");
    setFrom("");
    setTo("");
    setPlate("");
    setGate("all");
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <Box>
        <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
          <AssessmentIcon sx={{ color: GOLD }} />
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Reports & Export
          </Typography>
        </Stack>
        <Typography variant="body2" color="text.secondary">
          Filtered gate event reports with one-click CSV download and print-friendly view.
        </Typography>
      </Box>

      {/* Filter bar */}
      <Paper
        sx={{
          p: 2,
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
          bgcolor: "background.paper",
        }}
      >
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          alignItems={{ md: "center" }}
          flexWrap="wrap"
          useFlexGap
        >
          <FilterAltIcon sx={{ color: "text.secondary" }} />
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel id="range-label">Date range</InputLabel>
            <Select
              labelId="range-label"
              label="Date range"
              value={range}
              onChange={(e) => setRange(e.target.value as Range)}
            >
              <MenuItem value="today">Today</MenuItem>
              <MenuItem value="yesterday">Yesterday</MenuItem>
              <MenuItem value="last7">Last 7 days</MenuItem>
              <MenuItem value="last30">Last 30 days</MenuItem>
              <MenuItem value="custom">Custom…</MenuItem>
            </Select>
          </FormControl>

          {range === "custom" && (
            <>
              <TextField
                label="From"
                type="date"
                size="small"
                InputLabelProps={{ shrink: true }}
                value={from}
                onChange={(e) => setFrom(e.target.value)}
                sx={{ minWidth: 160 }}
              />
              <TextField
                label="To"
                type="date"
                size="small"
                InputLabelProps={{ shrink: true }}
                value={to}
                onChange={(e) => setTo(e.target.value)}
                sx={{ minWidth: 160 }}
              />
            </>
          )}

          <TextField
            label="Plate filter"
            size="small"
            value={plate}
            onChange={(e) => setPlate(e.target.value.toUpperCase())}
            placeholder="HR26BF1234"
            sx={{ minWidth: 180 }}
          />

          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel id="gate-label">Gate type</InputLabel>
            <Select
              labelId="gate-label"
              label="Gate type"
              value={gate}
              onChange={(e) => setGate(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="entry">Entry</MenuItem>
              <MenuItem value="exit">Exit</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ flex: 1 }} />

          <Tooltip title="Reset all filters" arrow>
            <Button
              startIcon={<RestartAltIcon />}
              variant="outlined"
              onClick={handleReset}
              size="small"
            >
              Reset
            </Button>
          </Tooltip>
          <Button
            startIcon={<PrintIcon />}
            variant="outlined"
            onClick={handlePrint}
            size="small"
            sx={{ "@media print": { display: "none" } }}
          >
            Print
          </Button>
          <Button
            startIcon={exporting ? <CircularProgress size={14} /> : <DownloadIcon />}
            variant="contained"
            onClick={handleDownload}
            disabled={exporting || total === 0}
            size="small"
            sx={{
              bgcolor: GOLD,
              color: "#000",
              "&:hover": { bgcolor: "#FFCB00" },
            }}
          >
            {exporting ? "Preparing…" : `Download CSV${total ? ` (${total})` : ""}`}
          </Button>
        </Stack>
      </Paper>

      {/* Summary cards */}
      <Grid container spacing={2}>
        <Grid item xs={6} md={3}>
          <StatCard
            label="Total events"
            value={isLoading ? "…" : total.toLocaleString()}
            color={GOLD}
            icon={<DirectionsBusIcon />}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <StatCard
            label="Entries"
            value={isLoading ? "…" : summary.entries.toLocaleString()}
            color={theme.palette.success.main}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <StatCard
            label="Exits"
            value={isLoading ? "…" : summary.exits.toLocaleString()}
            color={theme.palette.info.main}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <StatCard
            label="Anomalies"
            value={isLoading ? "…" : summary.anomalies.toLocaleString()}
            color={summary.anomalies > 0 ? theme.palette.warning.main : theme.palette.text.secondary}
          />
        </Grid>
      </Grid>

      {/* Result table */}
      <Paper
        sx={{
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
          overflow: "hidden",
          bgcolor: "background.paper",
        }}
      >
        {isLoading ? (
          <Box sx={{ p: 4, textAlign: "center" }}>
            <CircularProgress size={24} />
          </Box>
        ) : isError ? (
          <Alert severity="error" sx={{ m: 2 }}>
            Failed to load events.
            <Button size="small" sx={{ ml: 1 }} onClick={() => refetch()}>
              Retry
            </Button>
          </Alert>
        ) : rows.length === 0 ? (
          <Box sx={{ p: 4, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              No events found in the selected range.
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: "rgba(255,215,0,0.04)" }}>
                  <TableCell sx={{ fontWeight: 700 }}>When</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Plate</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Route</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Camera</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Gate</TableCell>
                  <TableCell sx={{ fontWeight: 700 }} align="right">Conf.</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Review</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.id} hover>
                    <TableCell sx={{ whiteSpace: "nowrap", fontSize: 12 }}>
                      {formatDateTime(r.timestamp)}
                    </TableCell>
                    <TableCell sx={{ fontFamily: "monospace", fontWeight: 700 }}>
                      {r.plate_number}
                    </TableCell>
                    <TableCell>
                      {r.route_number ? (
                        <Chip label={r.route_number} size="small" sx={{ bgcolor: "rgba(255,215,0,0.12)", color: GOLD, fontWeight: 700 }} />
                      ) : (
                        <Typography variant="caption" color="text.disabled">—</Typography>
                      )}
                    </TableCell>
                    <TableCell sx={{ fontSize: 12 }}>{r.camera_name}</TableCell>
                    <TableCell>
                      <Chip
                        label={r.gate_type}
                        size="small"
                        color={r.gate_type === "entry" ? "success" : "info"}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: "monospace" }}>
                      {Math.round((r.confidence || 0) * 100)}%
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={r.review_status}
                        size="small"
                        color={
                          r.review_status === "approved"
                            ? "success"
                            : r.review_status === "rejected"
                            ? "error"
                            : r.review_status === "corrected"
                            ? "info"
                            : "warning"
                        }
                        variant="outlined"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>

      <Typography variant="caption" color="text.disabled" sx={{ "@media print": { display: "none" } }}>
        Showing first {rows.length} of {total} events. Download CSV to get every row in the selected range.
      </Typography>
    </Box>
  );
}
