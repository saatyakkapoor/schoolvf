import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import BlockIcon from "@mui/icons-material/Block";
import PeopleIcon from "@mui/icons-material/People";
import {
  Alert as MuiAlert,
  Box,
  Button,
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
import { useState } from "react";

import { getUsers, createUser, updateUser, deleteUser } from "../api/users";
import { getApiErrorMessage } from "../utils/errors";
import type { User, CreateUserPayload, UpdateUserPayload } from "../types";

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

const roleColor = (role: string): "error" | "primary" | "default" => {
  switch (role) {
    case "admin":
      return "error";
    case "operator":
      return "primary";
    default:
      return "default";
  }
};

const formatDate = (iso: string | null) => {
  if (!iso) return "--";
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
};

interface UserFormState {
  username: string;
  password: string;
  display_name: string;
  role: string;
}

const emptyForm: UserFormState = {
  username: "",
  password: "",
  display_name: "",
  role: "viewer",
};

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function UsersPage() {
  const queryClient = useQueryClient();

  /* ---------- data ---------- */
  const { data: users, isLoading, isError, error } = useQuery({
    queryKey: ["users"],
    queryFn: getUsers,
  });

  /* ---------- dialog state ---------- */
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [form, setForm] = useState<UserFormState>({ ...emptyForm });

  /* ---------- confirm deactivate ---------- */
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const confirmUser = users?.find((u) => u.id === confirmId);

  /* ---------- snackbar ---------- */
  const [snack, setSnack] = useState<{ msg: string; severity: "success" | "error" } | null>(null);

  /* ---------- mutations ---------- */
  const createMut = useMutation({
    mutationFn: (p: CreateUserPayload) => createUser(p),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["users"] });
      closeDialog();
      setSnack({ msg: "User created successfully", severity: "success" });
    },
    onError: (e) => setSnack({ msg: getApiErrorMessage(e), severity: "error" }),
  });

  const updateMut = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateUserPayload }) =>
      updateUser(id, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["users"] });
      closeDialog();
      setSnack({ msg: "User updated successfully", severity: "success" });
    },
    onError: (e) => setSnack({ msg: getApiErrorMessage(e), severity: "error" }),
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteUser(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["users"] });
      setConfirmId(null);
      setSnack({ msg: "User deactivated", severity: "success" });
    },
    onError: (e) => {
      setConfirmId(null);
      setSnack({ msg: getApiErrorMessage(e), severity: "error" });
    },
  });

  /* ---------- dialog helpers ---------- */
  const openCreate = () => {
    setEditingUser(null);
    setForm({ ...emptyForm });
    setDialogOpen(true);
  };

  const openEdit = (u: User) => {
    setEditingUser(u);
    setForm({
      username: u.username,
      password: "",
      display_name: u.display_name,
      role: u.role,
    });
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setEditingUser(null);
  };

  const handleSubmit = () => {
    if (editingUser) {
      const payload: UpdateUserPayload = {
        display_name: form.display_name,
        role: form.role,
      };
      if (form.password) payload.password = form.password;
      updateMut.mutate({ id: editingUser.id, payload });
    } else {
      createMut.mutate({
        username: form.username.trim(),
        password: form.password,
        display_name: form.display_name.trim(),
        role: form.role,
      });
    }
  };

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
          <PeopleIcon color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h4" fontWeight={600}>
            User Management
          </Typography>
        </Stack>
        <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate}>
          Add User
        </Button>
      </Stack>

      {/* ---- Table ---- */}
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow sx={{ "& th": { fontWeight: 700, bgcolor: "grey.50" } }}>
              <TableCell>Username</TableCell>
              <TableCell>Display Name</TableCell>
              <TableCell>Role</TableCell>
              <TableCell align="center">Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Last Login</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users!.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center" sx={{ py: 4, color: "text.secondary" }}>
                  No users found. Click "Add User" to create the first one.
                </TableCell>
              </TableRow>
            )}
            {users!.map((u) => (
              <TableRow key={u.id} hover sx={{ opacity: u.is_active ? 1 : 0.55 }}>
                <TableCell sx={{ fontFamily: "monospace", fontWeight: 600 }}>{u.username}</TableCell>
                <TableCell>{u.display_name}</TableCell>
                <TableCell>
                  <Chip
                    label={u.role}
                    size="small"
                    color={roleColor(u.role)}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell align="center">
                  <Chip
                    label={u.is_active ? "Active" : "Inactive"}
                    size="small"
                    color={u.is_active ? "success" : "error"}
                    variant="filled"
                  />
                </TableCell>
                <TableCell>{formatDate(u.created_at)}</TableCell>
                <TableCell>{formatDate(u.last_login)}</TableCell>
                <TableCell align="right">
                  <Tooltip title="Edit">
                    <IconButton size="small" onClick={() => openEdit(u)}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {u.is_active && (
                    <Tooltip title="Deactivate">
                      <IconButton size="small" color="warning" onClick={() => setConfirmId(u.id)}>
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

      {/* ---- Add / Edit Dialog ---- */}
      <Dialog open={dialogOpen} onClose={closeDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{editingUser ? "Edit User" : "Add New User"}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} mt={1}>
            <TextField
              label="Username"
              required
              fullWidth
              disabled={!!editingUser}
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              placeholder="e.g. john.doe"
            />
            <TextField
              label={editingUser ? "Password (leave blank to keep current)" : "Password"}
              required={!editingUser}
              fullWidth
              type="password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
            />
            <TextField
              label="Display Name"
              required
              fullWidth
              value={form.display_name}
              onChange={(e) => setForm({ ...form, display_name: e.target.value })}
              placeholder="e.g. John Doe"
            />
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                label="Role"
                value={form.role}
                onChange={(e) => setForm({ ...form, role: e.target.value })}
              >
                <MenuItem value="admin">Admin</MenuItem>
                <MenuItem value="operator">Operator</MenuItem>
                <MenuItem value="viewer">Viewer</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={
              !form.display_name ||
              (!editingUser && (!form.username || !form.password)) ||
              isMutating
            }
          >
            {isMutating ? <CircularProgress size={20} /> : editingUser ? "Update" : "Create"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ---- Deactivate Confirm Dialog ---- */}
      <Dialog open={!!confirmId} onClose={() => setConfirmId(null)} maxWidth="xs">
        <DialogTitle>Deactivate User?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to deactivate user{" "}
            <strong>{confirmUser?.username}</strong>? They will no longer be able to log in.
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
