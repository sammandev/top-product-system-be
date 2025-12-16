# Scripts Reference Guide

All commands assume you are in the repository root (`backend_fastapi`) with the virtualenv activated. Replace sample paths as needed for your environment.

| Script | Purpose | Typical Usage |
| --- | --- | --- |
| `api_key_test.py` | Verifies API key / admin key enforcement by toggling environment variables and exercising `/api/compare` and `/api/cleanup-uploads`. | `uv run python scripts/api_key_test.py` |
| `bootstrap_rbac.py` | Bootstraps default permissions, roles, and seed users (`admin`, `analyst`, `viewer`). Supports password overrides and skipping user creation. | `uv run python scripts/bootstrap_rbac.py --admin-password S3cret` |
| `compare_formats.py` | Offline comparison of MasterControl vs DVT files; writes CSV and JSON summaries without hitting the API. | `uv run python scripts/compare_formats.py --master data/sample_data/000000000000018_MeasurePower_Lab_25G.csv --dvt data/sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv` |
| `compare_integration.py` | Uploads two files to a running API instance, triggers `/api/compare-download`, and saves the response CSV. | `uv run python scripts/compare_integration.py data/testdata/a.csv data/testdata/b.csv` |
| `convert_dvt_to_mc2.py` | Converts a DVT CSV/XLSX file into the MC2 CSV format on disk. | `uv run python scripts/convert_dvt_to_mc2.py --input data/sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv --output test_outputs/converted.csv` |
| `create_user.py` | Creates or updates a single local user and optionally grants admin rights. | `uv run python scripts/create_user.py --username alice --password pass123 --admin` |
| `debug_compare_call.py` | Generates temporary CSVs, uploads via `TestClient`, and prints compare diffs for local debugging. | `uv run python scripts/debug_compare_call.py` |
| `debug_compare_local.py` | Uses bundled sample files with `TestClient` to exercise `/api/compare-formats` without starting the server. | `uv run python scripts/debug_compare_local.py` |
| `debug_compare_request.py` | Sends raw HTTP requests (using `requests`) that mirror the comparison workflow against a running server. | `uv run python scripts/debug_compare_request.py` |
| `debug_compare_sample_data.py` | Runs comparisons on curated sample datasets to inspect output structure and regressions. | `uv run python scripts/debug_compare_sample_data.py` |
| `debug_duplicate_case_test.py` | Reproduces duplicate-case scenarios in compare logic to validate fixes. | `uv run python scripts/debug_duplicate_case_test.py` |
| `debug_duplicate_join.py` | Investigates duplicate join-key behaviour during compare operations. | `uv run python scripts/debug_duplicate_join.py` |
| `delete_compiled.py` | Removes a specific generated XLSX in `test_outputs/` (handy during repeated smoke runs). | `uv run python scripts/delete_compiled.py` |
| `delete_uploads_and_run_tests.py` | Clears the `uploads/` folder and launches `pytest` for a clean integration run. | `uv run python scripts/delete_uploads_and_run_tests.py` |
| `frontend_integration_sim.py` | Simulates the frontend workflow (upload → select → compare) against a running API. | `uv run python scripts/frontend_integration_sim.py` |
| `inspect_backend_app.py` | Imports the FastAPI app and prints route metadata for quick inspection. | `uv run python scripts/inspect_backend_app.py` |
| `integration_http.py` | Executes a suite of HTTP calls against the live API to verify parsing and compare endpoints. | `uv run python scripts/integration_http.py` |
| `integration_ui_simulate.py` | End-to-end UI simulation: upload, selection, comparison, and download via HTTP calls. | `uv run python scripts/integration_ui_simulate.py` |
| `post_compare.py` | Posts sample uploads and prints the JSON payload returned by `/api/compare`. | `uv run python scripts/post_compare.py` |
| `run_multi_dut_analysis.py` | Analyzes a multi-DUT MC2 file with optional spec JSON and outputs compiled XLSX. | `uv run python scripts/run_multi_dut_analysis.py --mc2 data/sample_data/2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv --spec data/sample_data_config/multi-dut_all_specs.json` |
| `run_multi_dut_analysis_alt.py` | Alternate pathway for multi-DUT analysis used for experimentation. | `uv run python scripts/run_multi_dut_analysis_alt.py --mc2 data/sample_data/2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv` |
| `run_real_comparison.py` | Full comparison of bundled sample MasterControl & DVT files; writes CSV & XLSX reports. | `uv run python scripts/run_real_comparison.py --margin-threshold 0.3` |
| `run_tests_with_server.ps1` | PowerShell helper that starts the API server, runs tests, and tears down. | `powershell.exe -File scripts/run_tests_with_server.ps1` |
| `smoke_generate_xlsx.py` | Creates a sample human-readable XLSX to verify formatting logic. | `uv run python scripts/smoke_generate_xlsx.py --margin-threshold 0.5` |
| `test_dvt_mc2_api.py` | Hits the FastAPI conversion endpoint using sample files for a quick smoke test. | `uv run python scripts/test_dvt_mc2_api.py` |

> Most scripts accept `-h/--help` for additional options. When targeting a running API, ensure the `ASTPARSER_BASE` (or similar) environment variable matches your host/port.

---

### Additional Tips

- **Toggle upload persistence:** set `UPLOAD_PERSIST=0` to force in-memory storage during debugging, or `1` to persist to disk.
- **Run the full test suite:** `uv run python -m pytest -v`.
- **Format, lint & lint-fix:** `make format`, `make lint` and `make fix` before submitting changes.
