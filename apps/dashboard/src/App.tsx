import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import RequireAuth from "./components/RequireAuth";
import MainLayout from "./layouts/MainLayout";
import AlertsPage from "./pages/AlertsPage";
import CamerasPage from "./pages/CamerasPage";
import DashboardPage from "./pages/DashboardPage";
import EventsPage from "./pages/EventsPage";
import LoginPage from "./pages/LoginPage";
import LivePage from "./pages/LivePage";
import PlateLookupPage from "./pages/PlateLookupPage";
import ReportsPage from "./pages/ReportsPage";
import TripsPage from "./pages/TripsPage";
import VehiclesPage from "./pages/VehiclesPage";
import UsersPage from "./pages/UsersPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <RequireAuth>
              <MainLayout />
            </RequireAuth>
          }
        >
          <Route index element={<DashboardPage />} />
          <Route path="live" element={<LivePage />} />
          <Route path="cameras" element={<CamerasPage />} />
          <Route path="trips" element={<TripsPage />} />
          <Route path="events" element={<EventsPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="plates" element={<PlateLookupPage />} />
          <Route path="vehicles" element={<VehiclesPage />} />
          <Route path="users" element={<UsersPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
