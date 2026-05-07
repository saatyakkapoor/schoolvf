import MenuIcon from "@mui/icons-material/Menu";
import DashboardIcon from "@mui/icons-material/Dashboard";
import VideocamIcon from "@mui/icons-material/Videocam";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import TimelineIcon from "@mui/icons-material/Timeline";
import EventIcon from "@mui/icons-material/Event";
import AssessmentIcon from "@mui/icons-material/Assessment";
import NotificationsIcon from "@mui/icons-material/Notifications";
import SearchIcon from "@mui/icons-material/Search";
import DirectionsBusIcon from "@mui/icons-material/DirectionsBus";
import PeopleIcon from "@mui/icons-material/People";
import {
  AppBar,
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link as RouterLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import { getMe } from "../api/auth";
import { useAuth } from "../context/AuthContext";
import { LiveWebSocketProvider } from "../hooks/useLiveWebSocket";

const DRAWER_WIDTH = 260;

const navItems: { to: string; label: string; icon: React.ReactNode }[] = [
  { to: "/", label: "Overview", icon: <DashboardIcon fontSize="small" /> },
  { to: "/live", label: "Live", icon: <VideocamIcon fontSize="small" /> },
  { to: "/cameras", label: "Cameras", icon: <CameraAltIcon fontSize="small" /> },
  { to: "/trips", label: "Trips", icon: <TimelineIcon fontSize="small" /> },
  { to: "/events", label: "Events", icon: <EventIcon fontSize="small" /> },
  { to: "/reports", label: "Reports", icon: <AssessmentIcon fontSize="small" /> },
  { to: "/alerts", label: "Alerts", icon: <NotificationsIcon fontSize="small" /> },
  { to: "/plates", label: "Plate lookup", icon: <SearchIcon fontSize="small" /> },
  { to: "/vehicles", label: "Fleet & Routes", icon: <DirectionsBusIcon fontSize="small" /> },
  { to: "/users", label: "Users", icon: <PeopleIcon fontSize="small" /> },
];

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  const location = useLocation();
  return (
    <List sx={{ px: 1 }}>
      {navItems.map((item) => {
        const selected =
          item.to === "/"
            ? location.pathname === "/"
            : location.pathname === item.to || location.pathname.startsWith(`${item.to}/`);
        return (
          <ListItemButton
            key={item.to}
            component={RouterLink}
            to={item.to}
            selected={selected}
            onClick={onNavigate}
            sx={{
              borderRadius: 1.5,
              mb: 0.5,
              py: 1,
              "&.Mui-selected": {
                bgcolor: "rgba(255,215,0,0.08)",
                borderLeft: "3px solid #FFD700",
                "&:hover": {
                  bgcolor: "rgba(255,215,0,0.12)",
                },
              },
              "&:hover": {
                bgcolor: "rgba(255,255,255,0.04)",
              },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 36,
                color: selected ? "#FFD700" : "#9AA0A6",
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontSize: "0.875rem",
                fontWeight: selected ? 600 : 400,
                color: selected ? "#E8EAED" : "#9AA0A6",
              }}
            />
          </ListItemButton>
        );
      })}
    </List>
  );
}

export default function MainLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { logout } = useAuth();
  const navigate = useNavigate();
  const { data: me } = useQuery({
    queryKey: ["auth", "me"],
    queryFn: getMe,
  });

  const handleLogout = () => {
    logout();
    navigate("/login", { replace: true });
  };

  const drawer = (
    <Box sx={{ pt: 2 }}>
      {/* School branding */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          px: 2,
          pb: 2,
        }}
      >
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 1,
            overflow: "hidden",
            mb: 1,
          }}
        >
          <img
            src="/logo.png"
            alt="TSRS Logo"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </Box>
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 700,
            color: "#E8EAED",
            fontSize: "0.8rem",
            letterSpacing: "0.05em",
          }}
        >
          TSRS Aravali
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: "#9AA0A6",
            fontSize: "0.7rem",
            letterSpacing: "0.08em",
          }}
        >
          Bus Monitor
        </Typography>
      </Box>
      <Divider sx={{ borderColor: "rgba(255,215,0,0.15)", mx: 2 }} />
      <NavList onNavigate={() => setMobileOpen(false)} />
    </Box>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <AppBar
        position="fixed"
        sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={() => setMobileOpen(true)}
            sx={{ mr: 2, display: { sm: "none" } }}
            aria-label="open menu"
          >
            <MenuIcon />
          </IconButton>
          <Box
            sx={{
              width: 28,
              height: 28,
              borderRadius: 0.75,
              overflow: "hidden",
              mr: 1.5,
              flexShrink: 0,
            }}
          >
            <img
              src="/logo.png"
              alt="TSRS"
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          </Box>
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              flexGrow: 0,
              fontSize: "1.05rem",
              fontWeight: 700,
              letterSpacing: "0.02em",
              borderBottom: "2px solid rgba(255,215,0,0.3)",
              pb: 0.3,
              display: "inline-block",
              width: "auto",
              mr: 2,
            }}
          >
            TSRS Bus Monitor
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Typography variant="body2" sx={{ opacity: 0.7, mr: 2, fontSize: "0.8rem" }}>
            {me?.username ?? "..."}
          </Typography>
          <Button
            color="inherit"
            onClick={handleLogout}
            sx={{
              fontSize: "0.8rem",
              opacity: 0.8,
              "&:hover": { opacity: 1, bgcolor: "rgba(255,255,255,0.05)" },
            }}
          >
            Log out
          </Button>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: DRAWER_WIDTH }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: "block", sm: "none" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: DRAWER_WIDTH },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: "none", sm: "block" },
            "& .MuiDrawer-paper": { boxSizing: "border-box", width: DRAWER_WIDTH },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
          mt: 8,
        }}
      >
        <LiveWebSocketProvider>
          <Outlet />
        </LiveWebSocketProvider>
      </Box>
    </Box>
  );
}
