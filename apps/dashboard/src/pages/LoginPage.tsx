import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";

import { loginRequest } from "../api/auth";
import { useAuth } from "../context/AuthContext";
import { getApiErrorMessage } from "../utils/errors";

export default function LoginPage() {
  const { setToken } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const expired = searchParams.get("expired") === "1";
  const from = (location.state as { from?: { pathname?: string } })?.from?.pathname ?? "/";

  const [username, setUsername] = useState("tsrs");
  const [password, setPassword] = useState("TSRS@2026");

  const loginMut = useMutation({
    mutationFn: () => loginRequest(username.trim(), password),
    onSuccess: (data) => {
      setToken(data.access_token);
      navigate(from, { replace: true });
    },
  });

  return (
    <Box
      minHeight="100vh"
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      sx={{
        background: "linear-gradient(145deg, #0a0e14 0%, #111820 40%, #0d1117 100%)",
        p: 2,
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "absolute",
          top: "-50%",
          left: "-50%",
          width: "200%",
          height: "200%",
          background:
            "radial-gradient(ellipse at center, rgba(255,215,0,0.03) 0%, transparent 50%)",
          pointerEvents: "none",
        },
      }}
    >
      {/* Logo */}
      <Box
        sx={{
          mb: 3,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Box
          sx={{
            width: 96,
            height: 96,
            borderRadius: 2,
            overflow: "hidden",
            mb: 2,
          }}
        >
          <img
            src="/logo.png"
            alt="TSRS Aravali Logo"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </Box>
        <Typography
          variant="h5"
          sx={{
            fontWeight: 700,
            color: "#E8EAED",
            letterSpacing: "0.02em",
            textAlign: "center",
          }}
        >
          The Shri Ram School Aravali
        </Typography>
        <Typography
          variant="subtitle2"
          sx={{
            color: "#FFD700",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            mt: 0.5,
          }}
        >
          Bus Monitoring System
        </Typography>
      </Box>

      {/* Login Card */}
      <Card
        sx={{
          maxWidth: 420,
          width: 1,
          borderLeft: "3px solid #FFD700",
          border: "1px solid rgba(255,255,255,0.06)",
          borderLeftColor: "#FFD700",
          borderLeftWidth: 3,
          borderLeftStyle: "solid",
          background: "linear-gradient(135deg, #111820 0%, #0d1117 100%)",
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Sign In
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 2, fontSize: "0.875rem" }}>
            Enter your credentials to access the dashboard.
          </Typography>

          {expired && (
            <Alert
              severity="warning"
              sx={{
                mb: 2,
                bgcolor: "rgba(255,183,77,0.08)",
                border: "1px solid rgba(255,183,77,0.2)",
                "& .MuiAlert-icon": { color: "#FFB74D" },
              }}
            >
              Your session expired. Please sign in again.
            </Alert>
          )}

          {loginMut.isError && (
            <Alert
              severity="error"
              sx={{
                mb: 2,
                bgcolor: "rgba(255,82,82,0.08)",
                border: "1px solid rgba(255,82,82,0.2)",
                "& .MuiAlert-icon": { color: "#ff5252" },
              }}
            >
              {getApiErrorMessage(loginMut.error)}
            </Alert>
          )}

          <Box
            component="form"
            onSubmit={(e) => {
              e.preventDefault();
              loginMut.mutate();
            }}
          >
            <TextField
              label="Username"
              fullWidth
              margin="normal"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              sx={{
                "& .MuiOutlinedInput-root": {
                  "&.Mui-focused fieldset": {
                    borderColor: "#FFD700",
                  },
                },
                "& .MuiInputLabel-root.Mui-focused": {
                  color: "#FFD700",
                },
              }}
            />
            <TextField
              label="Password"
              type="password"
              fullWidth
              margin="normal"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              sx={{
                "& .MuiOutlinedInput-root": {
                  "&.Mui-focused fieldset": {
                    borderColor: "#FFD700",
                  },
                },
                "& .MuiInputLabel-root.Mui-focused": {
                  color: "#FFD700",
                },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              size="large"
              sx={{
                mt: 3,
                py: 1.5,
                fontSize: "0.95rem",
                fontWeight: 700,
                letterSpacing: "0.03em",
              }}
              disabled={loginMut.isPending}
            >
              {loginMut.isPending ? "Signing in..." : "Sign In"}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Footer motto */}
      <Typography
        variant="caption"
        sx={{
          mt: 4,
          color: "rgba(255,215,0,0.35)",
          fontStyle: "italic",
          letterSpacing: "0.1em",
          fontSize: "0.8rem",
        }}
      >
        विद्या ददाति विनयम्
      </Typography>
    </Box>
  );
}
