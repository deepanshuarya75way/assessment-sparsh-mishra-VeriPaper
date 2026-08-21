import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    base: env.VITE_BASE_PATH || "/",
    build: {
      rollupOptions: {
        output: {
          // Manual chunking to separate heavy dependencies from core logic
          manualChunks: (id) => {
            if (id.includes("node_modules")) {
              if (id.includes("react") || id.includes("scheduler") || id.includes("prop-types")) {
                return "vendor-core";
              }
              if (id.includes("@sentry")) {
                return "vendor-sentry";
              }
              if (id.includes("axios") || id.includes("clsx") || id.includes("tailwind-merge")) {
                return "vendor-utils";
              }
              return "vendor-others";
            }
          }
        }
      },
      chunkSizeWarningLimit: 600,
      reportCompressedSize: true
    },
    server: {
      port: 5173
    }
  };
});
