import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const srcDir = fileURLToPath(new URL("./src", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": srcDir,
    },
  },
  server: {
    // Bind to all interfaces so other devices on the LAN can hit the dev
    // server at http://<host-ip>:3000. Override the proxy target with
    // VITE_DEV_API_TARGET when the API runs on another host.
    host: true,
    port: 3000,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
