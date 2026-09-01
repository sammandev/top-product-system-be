# Token Conservation Rules
- Be concise. Skip conversational filler, apologies, and restating the prompt.
- When suggesting changes, provide only the relevant snippet or unified diff rather than rewriting entire files.
- Favor brief inline explanations over lengthy markdown essays.

# Backend Copilot Instructions

## Project context

- This repository is the FastAPI backend for AST Tools.
- Python requires 3.13 or newer. Use `uv` for environments and dependencies.
- Application code lives under `src/app`; tests live under `tests`; schema changes live under `alembic`.
- The external DUT APIs are reachable only from the corporate intranet.

## Working rules

- Preserve the existing router -> service -> client layering and dependency-injection boundaries.
- Keep async I/O non-blocking. Reuse the existing pooled HTTP clients and caching helpers instead of creating unbounded clients per request.
- Intranet external API calls (DUT, iPLAS, SFISTSP) must use bounded timeouts and retries (maximum 2 retries with backoff). Map connection timeouts to explicit 504 Gateway Timeout or graceful degraded responses rather than unhandled server errors.
- Preserve API response shapes, authentication behavior, token handling, and existing error semantics unless the task explicitly changes the contract.
- Do not install PostgreSQL or Redis on the local machine just to validate a change. Use the configured environment, Docker services, or focused tests; ask before changing infrastructure assumptions.
- A local backend change does not update the deployed backend. Treat live-server verification and deployment as separate steps.
- Add or update focused tests when changing routes, services, external clients, caching, authentication, or migrations.
- Create Alembic migrations for database schema changes; do not edit production data or migration history by hand.
- Never expose bearer tokens, passwords, connection strings, or other secrets in source, logs, test output, or responses.

## Validation

Run the narrowest relevant checks first, then broaden when practical:

- `uv run ruff check .`
- `uv run python -m pytest -q tests/<relevant_test>.py`
- `uv run python -m pytest -q`
- `uv run python -m alembic check` when models or migrations change

Report the exact command and result. End substantial work with `State`, `Changed`, `Validation`, `Known issue`, and `Next action` so a new session can continue without rediscovering context.
