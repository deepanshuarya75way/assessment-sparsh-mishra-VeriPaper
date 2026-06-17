import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    // The FastAPI backend mounts the built frontend under /static, so
    // production builds must prefix asset references with /static/.
    base: env.VITE_BASE_PATH || "/",
    server: {
      port: 5173
    }
  };
});
