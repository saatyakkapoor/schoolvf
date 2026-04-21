import AddIcon from "@mui/icons-material/Add";
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import EditIcon from "@mui/icons-material/Edit";
import BlockIcon from "@mui/icons-material/Block";
import AltRouteIcon from "@mui/icons-material/AltRoute";
import PeopleIcon from "@mui/icons-material/People";
import LocalShippingIcon from "@mui/icons-material/LocalShipping";
import {
  Alert as MuiAlert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Snackbar,
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
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { getVehicles, createVehicle, updateVehicle, deleteVehicle } from "../api/vehicles";
import { getApiErrorMessage } from "../utils/errors";
import type { Vehicle, CreateVehiclePayload, UpdateVehiclePayload } from "../types";

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

const vehicleTypeColor = (t: string) => {
  switch (t.toLowerCase()) {
    case "bus":
      return "primary";
    case "van":
      return "secondary";
    default:
      return "default";
  }
};

const emptyForm: CreateVehiclePayload = {
  plate_number: "",
  vehicle_type: "bus",
  route_number: "",
  route_name: "",
  driver_name: "",
  driver_phone: "",
  capacity: 40,
};

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function VehiclesPage() {
  const queryClient = useQueryClient();

  /* ---------- data ---------- */
  const { data: vehicles, isLoading, isError, error } = useQuery({
    queryKey: ["vehicles"],
    queryFn: () => getVehicles(),
  });

  /* ---------- dialog state ---------- */
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingVehicle, setEditingVehicle] = useState<Vehicle | null>(null);
  const [form, setForm] = useState<CreateVehiclePayload>({ ...emptyForm });

  /* ---------- confirm deactivate ---------- */
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const confirmVehicle = vehicles?.find((v) => v.id === confirmId);

  /* ---------- snackbar ---------- */
  const [snack, setSnack] = useState<{ msg: string; severity: "success" | "error" } | null>(null);

  /* ---------- mutations ---------- */
  const createMut = useMutation({
    mutationFn: (p: CreateVehiclePayload) => createVehicle(p),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["vehicles"] });
      closeDialog();
      setSnack({ msg: "Vehicle registered successfully", severity: "success" });
    },
    onError: (e) => setSnack({ msg: getApiErrorMessage(e), severity: "error" }),
  });

  const updateMut = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateVehiclePayload }) =>
      updateVehicle(id, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["vehicles"] });
      closeDialog();
      setSnack({ msg: "Vehicle updated successfully", severity: "success" });
    },
    onError: (e) => setSnack({ msg: getApiErrorMessage(e), severity: "error" }),
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteVehicle(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["vehicles"] });
      setConfirmId(null);
      setSnack({ msg: "Vehicle deactivated", severity: "success" });
    },
    onError: (e) => {
      setConfirmId(null);
      setSnack({ msg: getApiErrorMessage(e), severity: "error" });
    },
  });

  /* ---------- dialog helpers ---------- */
  const openCreate = () => {
    setEditingVehicle(null);
    setForm({ ...emptyForm });
    setDialogOpen(true);
  };

  const openEdit = (v: Vehicle) => {
    setEditingVehicle(v);
    setForm({
      plate_number: v.plate_number,
      vehicle_type: v.vehicle_type,
      route_number: v.route_number,
      route_name: v.route_name,
      driver_name: v.driver_name,
      driver_phone: v.driver_phone,
      capacity: v.capacity,
    });
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setEditingVehicle(null);
  };

  const handleSubmit = () => {
    if (editingVehicle) {
      updateMut.mutate({ id: editingVehicle.id, payload: form as UpdateVehiclePayload });
    } else {
      createMut.mutate(form);
    }
  };

  /* ---------- stats ---------- */
  const stats = useMemo(() => {
    if (!vehicles) return { active: 0, routes: 0, capacity: 0, inactive: 0 };
    const active = vehicles.filter((v) => v.is_active);
    const routeSet = new Set(active.map((v) => v.route_number));
    return {
      active: active.length,
      routes: routeSet.size,
      capacity: active.reduce((s, v) => s + (v.capacity ?? 0), 0),
      inactive: vehicles.filter((v) => !v.is_active).length,
    };
  }, [vehicles]);

  /* ---------- loading / error ---------- */
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return <MuiAlert severity="error">{getApiErrorMessage(error)}</MuiAlert>;
  }

  const isMutating = createMut.isPending || updateMut.isPending;

  return (
    <Box>
      {/* ---- Header ---- */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={3}>
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <DirectionsBusIcon color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h4" fontWeight={600}>
            Fleet &amp; Routes
          </Typography>
        </Stack>
        <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate}>
          Register Vehicle
        </Button>
      </Stack>

      {/* ---- Stats ---- */}
      <Stack direction={{ xs: "column", sm: "row" }} spacing={2} mb={3}>
        <StatCard icon={<DirectionsBusIcon />} label="Active Vehicles" value={stats.active} color="#1976d2" />
        <StatCard icon={<AltRouteIcon />} label="Routes" value={stats.routes} color="#9c27b0" />
        <StatCard icon={<PeopleIcon />} label="Total Capacity" value={stats.capacity} color="#2e7d32" />
        <StatCard icon={<LocalShippingIcon />} label="Inactive" value={stats.inactive} color="#ed6c02" />
      </Stack>

      {/* ---- Table ---- */}
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow sx={{ "& th": { fontWeight: 700, bgcolor: "grey.50" } }}>
              <TableCell>Plate Number</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Route</TableCell>
              <TableCell>Route Name</TableCell>
              <TableCell>Driver</TableCell>
              <TableCell align="center">Capacity</TableCell>
              <TableCell align="center">Status</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {vehicles!.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} align="center" sx={{ py: 4, color: "text.secondary" }}>
                  No vehicles registered yet. Click "Register Vehicle" to get started.
                </TableCell>
              </TableRow>
            )}
            {vehicles!.map((v) => (
              <TableRow key={v.id} hover sx={{ opacity: v.is_active ? 1 : 0.55 }}>
                <TableCell sx={{ fontFamily: "monospace", fontWeight: 700, fontSize: 14 }}>
                  {v.plate_number}
                </TableCell>
                <TableCell>
                  <Chip
                    label={v.vehicle_type || "other"}
                    size="small"
                    color={vehicleTypeColor(v.vehicle_type) as "primary" | "secondary" | "default"}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip label={v.route_number} size="small" color="info" />
                </TableCell>
                <TableCell>{v.route_name || "--"}</TableCell>
                <TableCell>
                  {v.driver_name || "--"}
                  {v.driver_phone && (
                    <Typography variant="caption" display="block" color="text.secondary">
                      {v.driver_phone}
                    </Typography>
                  )}
                </TableCell>
                <TableCell align="center">{v.capacity ?? "--"}</TableCell>
                <TableCell align="center">
                  <Chip
                    label={v.is_active ? "Active" : "Inactive"}
                    size="small"
                    color={v.is_active ? "success" : "error"}
                    variant="filled"
                  />
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="Edit">
                    <IconButton size="small" onClick={() => openEdit(v)}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {v.is_active && (
                    <Tooltip title="Deactivate">
                      <IconButton size="small" color="warning" onClick={() => setConfirmId(v.id)}>
                        <BlockIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* ---- Register / Edit Dialog ---- */}
      <Dialog open={dialogOpen} onClose={closeDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{editingVehicle ? "Edit Vehicle" : "Register New Vehicle"}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} mt={1}>
            <TextField
              label="Plate Number"
              required
              fullWidth
              value={form.plate_number}
              onChange={(e) => setForm({ ...form, plate_number: e.target.value.toUpperCase() })}
              placeholder="e.g. HR-26-AB-1234"
              inputProps={{ style: { fontFamily: "monospace", fontWeight: 700 } }}
            />
            <FormControl fullWidth>
              <InputLabel>Vehicle Type</InputLabel>
              <Select
                label="Vehicle Type"
                value={form.vehicle_type ?? "bus"}
                onChange={(e) => setForm({ ...form, vehicle_type: e.target.value })}
              >
                <MenuItem value="bus">Bus</MenuItem>
                <MenuItem value="van">Van</MenuItem>
                <MenuItem value="car">Car</MenuItem>
              </Select>
            </FormControl>
            <Stack direction="row" spacing={2}>
              <TextField
                label="Route Number"
                required
                fullWidth
                value={form.route_number}
                onChange={(e) => setForm({ ...form, route_number: e.target.value })}
                placeholder="e.g. R-01"
              />
              <TextField
                label="Route Name"
                fullWidth
                value={form.route_name ?? ""}
                onChange={(e) => setForm({ ...form, route_name: e.target.value })}
                placeholder="e.g. Sector 56 - School"
              />
            </Stack>
            <Stack direction="row" spacing={2}>
              <TextField
                label="Driver Name"
                fullWidth
                value={form.driver_name ?? ""}
                onChange={(e) => setForm({ ...form, driver_name: e.target.value })}
              />
              <TextField
                label="Driver Phone"
                fullWidth
                value={form.driver_phone ?? ""}
                onChange={(e) => setForm({ ...form, driver_phone: e.target.value })}
                placeholder="+91 9876543210"
              />
            </Stack>
            <TextField
              label="Capacity"
              type="number"
              fullWidth
              value={form.capacity ?? ""}
              onChange={(e) => setForm({ ...form, capacity: Number(e.target.value) || 0 })}
              inputProps={{ min: 1 }}
            />
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={!form.plate_number || !form.route_number || isMutating}
          >
            {isMutating ? <CircularProgress size={20} /> : editingVehicle ? "Update" : "Register"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ---- Deactivate Confirm Dialog ---- */}
      <Dialog open={!!confirmId} onClose={() => setConfirmId(null)} maxWidth="xs">
        <DialogTitle>Deactivate Vehicle?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to deactivate{" "}
            <strong>{confirmVehicle?.plate_number}</strong>? The vehicle will no longer appear in
            active fleet listings.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmId(null)}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            disabled={deleteMut.isPending}
            onClick={() => confirmId && deleteMut.mutate(confirmId)}
          >
            {deleteMut.isPending ? <CircularProgress size={20} /> : "Deactivate"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ---- Snackbar ---- */}
      <Snackbar
        open={!!snack}
        autoHideDuration={4000}
        onClose={() => setSnack(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <MuiAlert
          onClose={() => setSnack(null)}
          severity={snack?.severity ?? "success"}
          variant="filled"
          elevation={6}
        >
          {snack?.msg}
        </MuiAlert>
      </Snackbar>
    </Box>
  );
}

/* ------------------------------------------------------------------ */
/*  Stat card                                                         */
/* ------------------------------------------------------------------ */

function StatCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <Card variant="outlined" sx={{ flex: 1, minWidth: 140 }}>
      <CardContent sx={{ display: "flex", alignItems: "center", gap: 2, py: 2, "&:last-child": { pb: 2 } }}>
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            bgcolor: `${color}14`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color,
          }}
        >
          {icon}
        </Box>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {label}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}
