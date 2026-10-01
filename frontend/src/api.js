import axios from "axios";

const configuredBase = (import.meta.env.VITE_API_BASE_URL || "").trim();
const fallbackBase = import.meta.env.DEV ? "http://localhost:8000/api" : "/api";

const api = axios.create({
  baseURL: configuredBase || fallbackBase,
});

export async function analyzePaper(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function analyzePaperAsync(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/analyze/async", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function pollAnalysisStatus(taskId) {
  const response = await api.get(`/analyze/${taskId}/status`);
  return response.data;
}

export async function pollUntilDone(taskId, onProgress, pollIntervalMs = 1000, timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const status = await pollAnalysisStatus(taskId);
    if (status.status === "done") return status;
    if (status.status === "error") throw new Error(status.error || "Analysis failed");
    if (onProgress && typeof onProgress === "function") onProgress(status);
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }
  throw new Error("Analysis timed out after " + timeoutMs / 1000 + " seconds");
}

export async function fetchHistory() {
  const response = await api.get("/history");
  return response.data;
}

export async function fetchHistoryItem(recordId) {
  const response = await api.get(`/history/${recordId}`);
  return response.data;
}

export async function fetchConfig() {
  const response = await api.get("/detector/config");
  return response.data;
}

export async function askReportQuestion(recordId, question){
  const response = await api.post(`/reports/${recordId}/chat`,{question});
  return response.data;
}

