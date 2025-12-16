"""Simple benchmarking harness for DUT Management endpoints.

Run with:
    uv run python scripts/benchmark_dut.py --iterations 5 --base-url http://localhost:8001 --token-file .cache/dut_tokens.json
    uv run python scripts\benchmark_dut.py --bearer-token "<access_token>" --iterations 2 --scenario-file temp\benchmark_scenarios.json

Scenario definitions can be supplied as a JSON file containing a list of request
specifications. Each item supports the following keys:
    name (str)    : friendly identifier used in output (required)
Bearer tokens can be supplied via --bearer-token, --token-file, or the
environment variable DUT_API_BEARER_TOKEN.  The script will also look for
.cache/dut_tokens.json by default.
    method (str)  : HTTP method such as "GET" or "POST" (default: GET)
    path (str)    : request path relative to the base URL (required)
    params (dict) : optional query parameters
    json (dict)   : optional JSON payload
    headers (dict): optional headers
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class BenchmarkScenario:
    name: str
    method: str = "GET"
    path: str = "/"
    params: dict[str, Any] | None = None
    json: Any = None
    headers: dict[str, str] | None = None
    iterations: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> BenchmarkScenario:
        try:
            name = data["name"]
            path = data["path"]
        except KeyError as exc:  # pragma: no cover - validated on load
            raise ValueError(f"Scenario definition missing required key: {exc}") from exc
        return cls(
            name=name,
            method=data.get("method", "GET").upper(),
            path=path,
            params=data.get("params"),
            json=data.get("json"),
            headers=data.get("headers"),
            iterations=data.get("iterations"),
        )


DEFAULT_SCENARIOS: tuple[BenchmarkScenario, ...] = (BenchmarkScenario(name="list_sites", method="GET", path="/api/dut/sites"),)


def _select_token_from_mapping(data: dict[str, Any]) -> str | None:
    # Prefer entries with explicit expiry, choose the furthest.
    if not data:
        return None
    best_token: tuple[str, Any] | None = None
    for _, entry in data.items():
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("access")
        expiry = entry.get("expiry")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        if best_token is None:
            best_token = (candidate, expiry)
            continue
        _, best_expiry = best_token
        if expiry and (not best_expiry or expiry > best_expiry):
            best_token = (candidate, expiry)
    if best_token is None:
        return None
    return best_token[0]


def resolve_bearer_token(token_arg: str | None, token_file: Path | None) -> str | None:
    if token_arg:
        return token_arg.strip()
    if token_file and token_file.exists():
        try:
            data = json.loads(token_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - CLI feedback
            print(f"Failed to read token file {token_file}: {exc}", file=sys.stderr)
        else:
            token = _select_token_from_mapping(data)
            if token:
                return token
    default_cache = Path(".cache/dut_tokens.json")
    if default_cache.exists():
        try:
            data = json.loads(default_cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            token = _select_token_from_mapping(data)
            if token:
                return token
    env_token = os.getenv("DUT_API_BEARER_TOKEN")
    if env_token:
        return env_token.strip()
    return None


def load_scenarios(path: Path | None) -> tuple[BenchmarkScenario, ...]:
    if path is None:
        return DEFAULT_SCENARIOS
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Iterable):  # pragma: no cover - guard rail
        raise ValueError("Scenario file must be a JSON array")
    scenarios: list[BenchmarkScenario] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each scenario entry must be a JSON object")
        scenarios.append(BenchmarkScenario.from_mapping(item))
    return tuple(scenarios)


def run_benchmark(
    client: httpx.Client,
    scenario: BenchmarkScenario,
    default_iterations: int,
    timeout: float,
    default_headers: dict[str, str],
) -> dict[str, Any]:
    elapsed_values: list[float] = []
    iterations = scenario.iterations or default_iterations
    for index in range(iterations):
        started = time.perf_counter()
        headers: dict[str, str] | None = None
        if default_headers or scenario.headers:
            headers = dict(default_headers)
            if scenario.headers:
                headers.update(scenario.headers)
        response = client.request(
            scenario.method,
            scenario.path,
            params=scenario.params,
            json=scenario.json,
            headers=headers,
            timeout=timeout,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "name": scenario.name,
                "success": False,
                "status": exc.response.status_code,
                "elapsed_ms": duration_ms,
                "iteration": index + 1,
                "detail": str(exc),
            }
        elapsed_values.append(duration_ms)
    mean_ms = statistics.mean(elapsed_values)
    p95 = statistics.quantiles(elapsed_values, n=20)[-1] if len(elapsed_values) > 1 else mean_ms
    return {
        "name": scenario.name,
        "success": True,
        "count": iterations,
        "avg_ms": round(mean_ms, 2),
        "min_ms": round(min(elapsed_values), 2),
        "max_ms": round(max(elapsed_values), 2),
        "p95_ms": round(p95, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DUT Management endpoints")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per scenario unless overridden",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        help="Optional path to JSON scenario definitions",
    )
    parser.add_argument(
        "--bearer-token",
        help="Explicit bearer token used for Authorization header",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Path to JSON cache file containing DUT tokens (defaults to .cache/dut_tokens.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenario_file)
    bearer_token = resolve_bearer_token(args.bearer_token, args.token_file)
    default_headers: dict[str, str] = {}
    if bearer_token:
        default_headers["Authorization"] = f"Bearer {bearer_token}"
    else:
        print(
            "Warning: No bearer token resolved. Requests to protected endpoints will fail with 401.",
            file=sys.stderr,
        )
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for scenario in scenarios:
            result = run_benchmark(
                client,
                scenario,
                args.iterations,
                args.timeout,
                default_headers,
            )
            results.append(result)
    for entry in results:
        if entry.get("success"):
            print(f"{entry['name']}: avg={entry['avg_ms']}ms min={entry['min_ms']}ms max={entry['max_ms']}ms p95={entry['p95_ms']}ms (n={entry['count']})")
        else:
            print(f"{entry['name']}: FAILED status={entry.get('status')} elapsed={entry.get('elapsed_ms'):.2f}ms iteration={entry.get('iteration')} detail={entry.get('detail')}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
