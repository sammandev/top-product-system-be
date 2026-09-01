---
applyTo: "**/*.py"
---

# Python Backend Rules

- Keep request handling in routers, business logic in services, and upstream communication in external clients. Follow nearby dependency-injection patterns.
- Keep async I/O non-blocking and reuse the existing pooled HTTP client helpers. Do not instantiate an unbounded `httpx.AsyncClient` per request.
- Preserve upstream error classification and user-facing error semantics. Do not replace useful interval, timeout, or authentication signals with a generic message.
- For external-client tests, patch the repository's client factory or dependency provider used by the code under test; do not patch a lower-level library when the production code no longer calls it directly.
- For authentication or token-dependent routes, test both dependency overrides and the expected token-bearing client attributes without using real credentials.
- Keep cache keys free of raw bearer tokens. Preserve negative-cache behavior and explicit cache invalidation when changing external API calls.
- Prefer focused async Pytest coverage for changed behavior, followed by Ruff and the full relevant test suite.
