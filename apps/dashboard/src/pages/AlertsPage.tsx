import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import FiberManualRecordIcon from "@mui/icons-material/FiberManualRecord";
import {
  Alert as MuiAlert,
  Badge,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Switch,
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
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { format, parseISO } from "date-fns";

import { getAlerts, resolveAlert } from "../api/alerts";
import { getApiErrorMessage } from "../utils/errors";
import type { AlertSeverity } from "../types";

const GOLD = "#FFD700";

function formatDateTime(iso: string): string {
  try {
    return format(parseISO(iso), "dd MMM yyyy, hh:mm:ss a");
  } catch {
    return iso;
  }
}

function severityColor(severity: string): "info" | "warning" | "error" {
  if (severity === "warning") return "warning";
  if (severity === "critical") return "error";
  return "info";
}

function severityDotColor(severity: string): string {
  if (severity === "critical") return "#f44336";
  if (severity === "warning") return "#ff9800";
  return "#2196f3";
}

function alertTypeLabel(type: string): string {
  return type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function AlertsPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [resolvedFilter, setResolvedFilter] = useState(false);

  // Resolve dialog state
  const [resolveDialogOpen, setResolveDialogOpen] = useState(false);
  const [resolveAlertId, setResolveAlertId] = useState<string | null>(null);
  const [resolveNote, setResolveNote] = useState("");

  const filters: Record<string, unknown> = {
    page: page + 1,
    page_size: pageSize,
  };
  if (severityFilter !== "all") filters.severity = severityFilter as AlertSeverity;
  if (!resolvedFilter) filters.resolved = false;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["alerts", filters],
    queryFn: () => getAlerts(filters as any),
    placeholderData: (prev) => prev,
  });

  const resolveMut = useMutation({
    mutationFn: ({ id, note }: { id: string; note: string }) =>
      resolveAlert(id, { resolved_by: "admin", resolution_note: note }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      setResolveDialogOpen(false);
      setResolveNote("");
      setResolveAlertId(null);
    },
  });

  const activeCount = data ? data.items.filter((a) => !a.resolved).length : 0;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <Badge badgeContent={activeCount} color="error" max={99}>
          <NotificationsActiveIcon sx={{ fontSize: 32, color: GOLD }} />
        </Badge>
        <Typography variant="h4" fontWeight={700} sx={{ color: "#fff" }}>
          Alerts & Anomalies
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
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Severity</InputLabel>
          <Select
            value={severityFilter}
            label="Severity"
            onChange={(e) => {
              setSeverityFilter(e.target.value);
              setPage(0);
            }}
          >
            <MenuItem value="all">All severities</MenuItem>
            <MenuItem value="info">Info</MenuItem>
            <MenuItem value="warning">Warning</MenuItem>
            <MenuItem value="critical">Critical</MenuItem>
          </Select>
        </FormControl>
        <FormControlLabel
          control={
            <Switch
              checked={resolvedFilter}
              onChange={(e) => {
                setResolvedFilter(e.target.checked);
                setPage(0);
              }}
              sx={{
                "& .MuiSwitch-switchBase.Mui-checked": { color: GOLD },
                "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track": { bgcolor: GOLD },
              }}
            />
          }
          label={
            <Typography variant="body2" color="text.secondary">
              Show resolved
            </Typography>
          }
        />
      </Paper>

      {/* Error */}
      {isError && (
        <MuiAlert severity="error" sx={{ mb: 2 }}>
          {getApiErrorMessage(error)}
        </MuiAlert>
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
          <CheckCircleOutlineIcon sx={{ fontSize: 64, color: "rgba(255,215,0,0.25)", mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No alerts to show
          </Typography>
          <Typography variant="body2" color="text.disabled" maxWidth={480} mx="auto">
            {resolvedFilter
              ? "No alerts match the current filters."
              : "All clear! No unresolved alerts at the moment. Toggle 'Show resolved' to view past alerts."}
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
                  <TableCell>Created</TableCell>
                  <TableCell>Plate</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.items.map((a) => (
                  <TableRow
                    key={a.id}
                    sx={{
                      "&:hover": { bgcolor: "rgba(255,215,0,0.04)" },
                      "& td": { borderBottom: "1px solid rgba(255,255,255,0.05)" },
                    }}
                  >
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {formatDateTime(a.created_at)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{ fontFamily: "monospace", fontWeight: 700, letterSpacing: 1, color: "#fff" }}
                      >
                        {a.plate_number}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {alertTypeLabel(a.alert_type)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={
                          <FiberManualRecordIcon
                            sx={{ fontSize: "10px !important", color: `${severityDotColor(a.severity)} !important` }}
                          />
                        }
                        label={a.severity.toUpperCase()}
                        color={severityColor(a.severity)}
                        variant="outlined"
                        size="small"
                        sx={{ fontWeight: 700, fontSize: 11 }}
                      />
                    </TableCell>
                    <TableCell>
                      {a.resolved ? (
                        <Chip
                          label="RESOLVED"
                          color="success"
                          variant="outlined"
                          size="small"
                          sx={{ fontWeight: 600, fontSize: 11 }}
                        />
                      ) : (
                        <Chip
                          label="ACTIVE"
                          color="error"
                          size="small"
                          sx={{ fontWeight: 600, fontSize: 11 }}
                        />
                      )}
                    </TableCell>
                    <TableCell align="right">
                      {!a.resolved && (
                        <Button
                          size="small"
                          variant="outlined"
                          sx={{
                            borderColor: GOLD,
                            color: GOLD,
                            fontSize: 11,
                            "&:hover": { borderColor: GOLD, bgcolor: "rgba(255,215,0,0.08)" },
                          }}
                          onClick={() => {
                            setResolveAlertId(a.id);
                            setResolveDialogOpen(true);
                          }}
                        >
                          Resolve
                        </Button>
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

      {/* Resolve Alert Dialog */}
      <Dialog
        open={resolveDialogOpen}
        onClose={() => {
          setResolveDialogOpen(false);
          setResolveNote("");
        }}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#1a1a2e",
            border: "1px solid rgba(255,215,0,0.15)",
          },
        }}
      >
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <WarningAmberIcon sx={{ color: GOLD }} />
          Resolve Alert
        </DialogTitle>
        <DialogContent>
          <TextField
            label="Resolution note"
            fullWidth
            multiline
            minRows={3}
            value={resolveNote}
            onChange={(e) => setResolveNote(e.target.value)}
            placeholder="Describe how this alert was resolved..."
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={() => {
              setResolveDialogOpen(false);
              setResolveNote("");
            }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={!resolveNote.trim() || resolveMut.isPending}
            onClick={() => {
              if (resolveAlertId) {
                resolveMut.mutate({ id: resolveAlertId, note: resolveNote.trim() });
              }
            }}
            sx={{
              bgcolor: GOLD,
              color: "#000",
              fontWeight: 700,
              "&:hover": { bgcolor: "#e6c200" },
            }}
          >
            {resolveMut.isPending ? "Resolving..." : "Resolve"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
