import axios from "axios";

/**
 * Same-origin `/api` works behind the dashboard nginx proxy or Vite dev proxy.
 * Override with `VITE_API_BASE` (e.g. `http://127.0.0.1:8000/api`) if you run the UI without a proxy.
 */
const baseURL = import.meta.env.VITE_API_BASE ?? "/api";

const apiClient = axios.create({
  baseURL,
  timeout: 15_000,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("auth_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      const url = String(error.config?.url ?? "");
      if (url.includes("/auth/login")) {
        return Promise.reject(error);
      }
      localStorage.removeItem("auth_token");
      const path = window.location.pathname;
      if (!path.startsWith("/login")) {
        window.location.assign("/login?expired=1");
      }
    }
    return Promise.reject(error);
  },
);

export default apiClient;
