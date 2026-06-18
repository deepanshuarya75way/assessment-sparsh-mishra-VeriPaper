/**
 * Download PDF report from the backend
 */
export function downloadPDF(reportPath) {
  if (!reportPath) {
    alert("No PDF report available.");
    return;
  }

  const configuredBase = (import.meta.env.VITE_API_BASE_URL || "").trim();
  const apiOrigin = configuredBase ? new URL(configuredBase, window.location.origin).origin : window.location.origin;
  const absoluteReportPath = reportPath.startsWith("http") ? reportPath : `${apiOrigin}${reportPath}`;

  const filename = `VeriPaper_Report_${new Date().toISOString().split("T")[0]}.pdf`;
  const link = document.createElement("a");
  link.href = absoluteReportPath;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Export analysis results as CSV
 */
export function downloadCSV(result) {
  const headers = ["Metric", "Score", "Details"];

  const rows = [
    ["Overall Research Credibility", `${result.overall_research_credibility}%`, result.verdict_detail || ""],
    ["Plagiarism / Similarity", `${Math.round(result.plagiarism?.score ?? 0)}%`, result.plagiarism?.summary || ""],
    ["AI-Generated Probability", `${Math.round(result.ai_detection?.ai_probability ?? 0)}%`, `Confidence: ${result.ai_detection?.confidence || ""}`],
    ["Citation Validity", `${Math.round(result.citation?.validity_score ?? 0)}%`, result.citation?.summary || ""],
    ["Statistical Risk", `${Math.round(result.statistics?.risk_score ?? 0)}%`, result.statistics?.summary || ""],
    ["Writing Standards", `${Math.round(result.writing?.score ?? 0)}%`, result.writing?.grade || ""],
    ["", "", ""],
    ["Word Count", result.word_count || "", ""],
    ["Sections Detected", result.section_count || "", ""],
    ["Duplicate Paragraphs (self-plagiarism)", result.plagiarism?.duplicate_paragraphs ?? 0, ""],
    ["Valid DOIs", result.citation?.valid_dois ?? 0, `of ${result.citation?.total_dois ?? 0} total`],
    ["Invalid DOIs Found", (result.citation?.invalid_dois || []).length, (result.citation?.invalid_dois || []).join("; ")],
    ["Top Plagiarism Match", result.plagiarism?.matches?.[0]?.similarity ? `${Math.round(result.plagiarism.matches[0].similarity)}%` : "N/A", result.plagiarism?.matches?.[0]?.title || ""],
  ];

  const csv = [
    headers.join(","),
    ...rows.map(row => 
      row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(",")
    )
  ].join("\n");

  const filename = `VeriPaper_Analysis_${new Date().toISOString().split("T")[0]}.csv`;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

/**
 * Export analysis results as JSON
 */
export function downloadJSON(result) {
  const filename = `VeriPaper_Analysis_${new Date().toISOString().split("T")[0]}.json`;
  const json = JSON.stringify(result, null, 2);
  const blob = new Blob([json], { type: "application/json;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

/**
 * Save analysis result to localStorage history
 */
export function saveToHistory(result) {
  try {
    const history = getHistory();
    const entry = {
      ...result,
      timestamp: new Date().toISOString(),
      id: `analysis_${Date.now()}`
    };
    const updated = [entry, ...history].slice(0, 10); // history guaranteed an array by getHistory()
    localStorage.setItem("veripaper_history", JSON.stringify(updated));
  } catch (e) {
    console.error("Failed to save history:", e);
  }
}

/**
 * Get all history entries from localStorage
 */
export function getHistory() {
  try {
    const stored = localStorage.getItem("veripaper_history");
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.error("Failed to parse history:", e);
    return [];
  }
}

/**
 * Clear entire history from localStorage
 */
export function clearHistory() {
  try {
    localStorage.removeItem("veripaper_history");
  } catch (e) {
    console.error("Failed to clear history:", e);
  }
}

