import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build goes into the Python package (src/matchminer_ai/apps/dashboard/static), which FastAPI serves at "/".
// `pnpm dev` serves the frontend on :5173 and proxies /api to a running `mmai-dashboard`.
const API = process.env.MMAI_DASH_API_URL ?? "http://127.0.0.1:8781";

export default defineConfig({
  base: "/",
  plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: { outDir: path.resolve(import.meta.dirname, "../../src/matchminer_ai/apps/dashboard/static"), emptyOutDir: true },
  server: { host: "127.0.0.1", proxy: { "/api": { target: API } } },
});
