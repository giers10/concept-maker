import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const host = process.env.TAURI_DEV_HOST;
  return {
    plugins: [react()],
    clearScreen: false,
    server: {
      port: 5173,
      strictPort: true,
      host: host || false,
      hmr: host
        ? {
            protocol: "ws",
            host,
            port: 5173,
          }
        : undefined,
    },
    build: {
      target: "esnext",
      minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
      sourcemap: !!process.env.TAURI_DEBUG,
    },
  };
});
