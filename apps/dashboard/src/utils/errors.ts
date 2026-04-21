import axios from "axios";

/** Human-readable message for failed API calls (network, 4xx, 5xx). */
export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      const status = error.response.status;
      const d = error.response.data as { detail?: string | unknown } | undefined;
      if (typeof d?.detail === "string") return d.detail;
      if (Array.isArray(d?.detail)) return JSON.stringify(d.detail);
      if (status === 401) return "Not signed in or session expired.";
      if (status === 502) {
        return "Bad gateway (502): nginx could not reach the API. Rebuild the dashboard image (nginx fix), run docker compose ps to ensure api is healthy, then retry.";
      }
      if (status === 503) {
        return "Service unavailable (503): the API may still be starting. Wait and retry.";
      }
      return `Request failed (${status})`;
    }
    if (error.code === "ERR_NETWORK" || error.message === "Network Error") {
      return "Cannot reach the API. If you use “npm run dev”, start the backend on port 8000 (or set VITE_API_BASE). In Docker, open the dashboard port and ensure the api service is up.";
    }
    return error.message || "Request failed";
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong";
}
