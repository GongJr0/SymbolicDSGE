import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { pyrightLspPlugin } from "./plugins/pyrightLsp";

export default defineConfig({
  plugins: [react(), pyrightLspPlugin()],
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
});
