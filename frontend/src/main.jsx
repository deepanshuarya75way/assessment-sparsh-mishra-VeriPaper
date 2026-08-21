import React from "react";
import ReactDOM from "react-dom/client";
import "./sentry.js"; // must be the first app import — initializes error tracking
import App from "./App.jsx";
import "./index.css";
import Sentry from "./sentry.js";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Sentry.ErrorBoundary fallback={<AppCrashFallback />}>
      <App />
    </Sentry.ErrorBoundary>
  </React.StrictMode>
);

// Friendly fallback if an unrecoverable error renders the whole app unusable.
// Sentry reports the crash automatically; this gives the user a recovery path.
function AppCrashFallback({ error }) {
  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 max-w-md text-center">
        <h1 className="text-xl font-bold text-white mb-2">Something went wrong</h1>
        <p className="text-slate-300 mb-6 text-sm">
          VeriPaper hit an unexpected error and we have already reported it.
          Try refreshing the page — your analysis history is saved locally.
        </p>
        <button
          onClick={() => window.location.reload()}
          className="px-5 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition-colors"
        >
          Refresh page
        </button>
      </div>
    </div>
  );
}
