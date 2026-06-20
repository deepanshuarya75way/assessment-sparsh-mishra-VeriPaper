import * as Sentry from "@sentry/react";

// Production error tracking — Sentry free tier (veripaper org, EU region).
// Error monitoring only: no tracing, no session replay, to stay within the
// free-plan budget (5,000 errors/month after the 14-day trial).
Sentry.init({
  dsn: "https://c5c92309ea775dd3d9fee496a8c05312@o4511942207799296.ingest.de.sentry.io/4511942214287440",
  environment: "production",
  tracesSampleRate: 0,
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 0,
  // Do not capture full HTTP request/response bodies to keep payloads small.
  beforeSend(event) {
    if (event.request) {
      event.request.data = undefined;
    }
    return event;
  },
});

export default Sentry;
