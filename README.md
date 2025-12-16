# AST Parser FastAPI Backend

Modern FastAPI service for uploading and analysing wireless test data, comparing MasterControl vs DVT formats, converting reports, and brokering access to the internal DUT Management API.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-239%20passing-brightgreen.svg)](./tests/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 📑 Table of Contents

- [AST Parser FastAPI Backend](#ast-tools-fastapi-backend)
  - [📑 Table of Contents](#-table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Create and activate a virtual environment (via uv)](#2-create-and-activate-a-virtual-environment-via-uv)
    - [3. Install dependencies](#3-install-dependencies)
      - [3.5 Updates dependencies](#35-updates-dependencies)
    - [4. Configure environment](#4-configure-environment)
    - [5. Run database migrations (SQLAlchemy metadata create)](#5-run-database-migrations-sqlalchemy-metadata-create)
    - [6. Seed default users, roles, and permissions](#6-seed-default-users-roles-and-permissions)
    - [7. Run the development server](#7-run-the-development-server)
    - [8. Run the automated test suite](#8-run-the-automated-test-suite)
  - [Project Structure](#project-structure)
  - [Common Commands](#common-commands)
  - [Tech Stack \& Updating Dependencies](#tech-stack--updating-dependencies)
    - [Upgrade workflow](#upgrade-workflow)
  - [Database \& Migration Notes](#database--migration-notes)
    - [Initial Setup](#initial-setup)
    - [Creating New Migrations](#creating-new-migrations)
    - [Migration Commands](#migration-commands)
      - [Alembic workflow tips](#alembic-workflow-tips)
    - [Seeding Data](#seeding-data)
    - [RBAC quickstart](#rbac-quickstart)
  - [📚 API Endpoint Reference](#-api-endpoint-reference)
  - [1. 🏠 Root \& Documentation Endpoints](#1--root--documentation-endpoints)
    - [`GET /`](#get-)
    - [`GET /swagger`](#get-swagger)
    - [`GET /redoc`](#get-redoc)
  - [2. 🔐 Authentication Endpoints (`/api/auth`)](#2--authentication-endpoints-apiauth)
    - [`POST /api/auth/login`](#post-apiauthlogin)
    - [`POST /api/auth/external-login`](#post-apiauthexternal-login)
    - [`GET /api/auth/me`](#get-apiauthme)
    - [`POST /api/auth/token/refresh`](#post-apiauthtokenrefresh)
    - [`POST /api/auth/users/create`](#post-apiauthuserscreate)
    - [`POST /api/auth/users/change-password`](#post-apiauthuserschange-password)
    - [`DELETE /api/auth/users/{username}`](#delete-apiauthusersusername)
    - [`POST /api/auth/users/revoke`](#post-apiauthusersrevoke)
    - [`POST /api/auth/external-token`](#post-apiauthexternal-token)
  - [3. 👥 RBAC Endpoints (`/api/rbac`)](#3--rbac-endpoints-apirbac)
    - [`POST /api/rbac/roles`](#post-apirbacroles)
    - [`GET /api/rbac/roles`](#get-apirbacroles)
    - [`DELETE /api/rbac/roles/{role_id}`](#delete-apirbacrolesrole_id)
    - [`POST /api/rbac/permissions`](#post-apirbacpermissions)
    - [`GET /api/rbac/permissions`](#get-apirbacpermissions)
    - [`DELETE /api/rbac/permissions/{perm_id}`](#delete-apirbacpermissionsperm_id)
    - [`POST /api/rbac/roles/{role_id}/grant`](#post-apirbacrolesrole_idgrant)
    - [`POST /api/rbac/roles/{role_id}/revoke`](#post-apirbacrolesrole_idrevoke)
    - [`POST /api/rbac/users/{user_id}/assign-role`](#post-apirbacusersuser_idassign-role)
    - [`POST /api/rbac/users/{user_id}/remove-role`](#post-apirbacusersuser_idremove-role)
    - [`GET /api/rbac/demo/needs-read`](#get-apirbacdemoneeds-read)
  - [4. 📁 Upload \& Parsing Endpoints (`/api/upload-preview`, `/api/parse`, `/api/cleanup-uploads`)](#4--upload--parsing-endpoints-apiupload-preview-apiparse-apicleanup-uploads)
    - [`POST /api/upload-preview`](#post-apiupload-preview)
    - [`POST /api/parse`](#post-apiparse)
    - [`POST /api/parse-download`](#post-apiparse-download)
    - [`POST /api/cleanup-uploads`](#post-apicleanup-uploads)
  - [5. ⚖️ Comparison Endpoints (`/api/compare`)](#5-️-comparison-endpoints-apicompare)
    - [`POST /api/compare`](#post-apicompare)
    - [`POST /api/compare-download`](#post-apicompare-download)
  - [6. 📊 Format Comparison Endpoint (`/api/compare-formats`)](#6--format-comparison-endpoint-apicompare-formats)
    - [`POST /api/compare-formats`](#post-apicompare-formats)
  - [7. 🔄 DVT to MC2 Conversion (`/api/convert-dvt-to-mc2`)](#7--dvt-to-mc2-conversion-apiconvert-dvt-to-mc2)
    - [`POST /api/convert-dvt-to-mc2`](#post-apiconvert-dvt-to-mc2)
  - [8. 📈 Multi-DUT Analysis (`/api/analyze-multi-dut`)](#8--multi-dut-analysis-apianalyze-multi-dut)
    - [`POST /api/analyze-multi-dut`](#post-apianalyze-multi-dut)
  - [9. 🌐 DUT Management \& External API (`/api/dut/*`)](#9--dut-management--external-api-apidut)
    - [Overview](#overview)
    - [9.1 Metadata Endpoints](#91-metadata-endpoints)
      - [`GET /api/dut/sites`](#get-apidutsites)
      - [`GET /api/dut/sites/{site_id}/models`](#get-apidutsitessite_idmodels)
      - [`GET /api/dut/models/{model_id}/stations`](#get-apidutmodelsmodel_idstations)
    - [9.2 Test Record Endpoints](#92-test-record-endpoints)
      - [`GET /api/dut/records/{station_id}/{dut_id}`](#get-apidutrecordsstation_iddut_id)
      - [`POST /api/dut/test-items/latest/batch`](#post-apiduttest-itemslatestbatch)
    - [9.3 History Endpoints](#93-history-endpoints)
      - [`GET /api/dut/history/isns`](#get-apiduthistoryisns)
      - [`GET /api/dut/history/identifiers`](#get-apiduthistoryidentifiers)
      - [`GET /api/dut/history/progression`](#get-apiduthistoryprogression)
      - [`GET /api/dut/history/results`](#get-apiduthistoryresults)
    - [9.4 Summary \& Analytics](#94-summary--analytics)
      - [`GET /api/dut/summary`](#get-apidutsummary)
      - [`POST /api/dut/stations/{station_id}/top-products`](#post-apidutstationsstation_idtop-products)
      - [`POST /api/dut/pa-test-items/adjusted-power`](#post-apidutpa-test-itemsadjusted-power)
    - [Overview](#overview-1)
    - [Scoring Pipeline](#scoring-pipeline)
    - [Category Detection](#category-detection)
    - [Default Scoring (Standard Measurements)](#default-scoring-standard-measurements)
    - [EVM Scoring (Error Vector Magnitude)](#evm-scoring-error-vector-magnitude)
    - [PER Scoring (Packet Error Rate)](#per-scoring-packet-error-rate)
    - [Overall Score Calculation](#overall-score-calculation)
    - [Criteria File Format](#criteria-file-format)
    - [Response Format](#response-format)
    - [Practical Usage Tips](#practical-usage-tips)
    - [Summary](#summary)
  - [Testing \& Quality Assurance](#testing--quality-assurance)
  - [Troubleshooting](#troubleshooting)
  - [Contributing Guidelines](#contributing-guidelines)
  - [License](#license)

---

## Prerequisites

- **Python ≥ 3.13**
- **uv** package/virtualenv manager (`pip install uv` if you do not already have it)
- **PostgreSQL 14+** (default DSN in `.env` points at `postgres:5432`)
- **Redis** *(optional)* — only required when caching DUT tokens
- **Git** and **Make** (or run the equivalent `uv` commands manually)

> The external DUT APIs are reachable only from the corporate intranet.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/backend_fastapi.git
cd backend_fastapi
```

### 2. Create and activate a virtual environment (via uv)

```bash
uv venv .venv
.\.venv\Scripts\activate   # PowerShell
# source .venv/bin/activate   # Linux / macOS
```

### 3. Install dependencies

```bash
make install              # wraps: uv pip install -e .
# or manually:
# uv pip install -e .
```

#### 3.5 Updates dependencies

Run these commands on your terminal (make sure virtual environment activated)

**Check the latest version of installed packages:**

```bash
uv pip list --outdated
```

**Upgrade all locked packages**

```bash
uv lock --upgrade
```

The command upgrades all locked versions of the packages to the latest compatible versions based on the `pyproject.toml` file.

**Then synchronize the environment**

```bash
uv sync --check
uv sync
```

To sync the project's dependencies with the environment.

or

Update your project's dependencies using your `requirements.txt file`.

```bash
uv pip sync requirements.txt
uv sync
```

After updating the lockfile, you need to synchronize your virtual environment with the updated lockfile to install or upgrade the packages.

```bash
uv lock
```

**Check the compatibility of installed packages**

```bash
uv pip check
```

### 4. Configure environment

Copy `.env.example` (if available) or adjust `.env`:

```env
APP_NAME=Test123
DEBUG=True
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/test123
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=change_me
UPLOAD_PERSIST=1
DUT_API_BASE_URL=http://192.168.180.56:9001
```

The `ARGON2_*` settings feed directly into the `pwdlib` Argon2 hasher, so you can dial CPU and memory requirements per environment.
If `REDIS_URL` cannot be reached the service now falls back to an in-memory DUT token cache (tokens will be lost when the process restarts), so running a local Redis instance is recommended for persistent caching.

### 5. Run database migrations (SQLAlchemy metadata create)

```bash
# Create the schema and stamp migration history in one go
uv run python -c "from app.db.init_db import init_db; init_db()"

# or apply Alembic migrations directly
make upgrade
# uv run alembic upgrade head
```

### 6. Seed default users, roles, and permissions

```bash
# seeds permissions, roles, and optional default users (admin/analyst/viewer)
uv run python scripts/bootstrap_rbac.py \
  --admin-password "AdminPass!123" \
  --analyst-password "AnalystPass!123" \
  --viewer-password "ViewerPass!123"
```

Flags you can use:

- `--skip-default-users` – create permissions/roles only.
- `--admin-password`, `--analyst-password`, `--viewer-password` – override the defaults for seeded accounts.
- `--help` – display all options.

To add ad-hoc users later:

```bash
uv run python scripts/create_user.py --username qa_user --password "Str0ng!" 
uv run python scripts/create_user.py --username ops_admin --admin
```

Assign or remove roles via the RBAC API (requires an admin token):

```bash
# assign role id=3 to user id=7
curl -X POST http://127.0.0.1:8001/api/rbac/users/7/assign-role \
  -H "Authorization: Bearer <ADMIN_ACCESS_TOKEN>" \
  -F "role_id=3"

# remove the role
curl -X POST http://127.0.0.1:8001/api/rbac/users/7/remove-role \
  -H "Authorization: Bearer <ADMIN_ACCESS_TOKEN>" \
  -F "role_id=3"
```

### 7. Run the development server

```bash
make dev              # fastapi dev src/app/main.py --port 8001
# or production-style:
make start            # uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### 8. Run the automated test suite

```bash
make test             # uv run pytest -v
```

**Test Suite Status (December 2025)**:
- ✅ **239 passing** unit tests (98% coverage)
- ⏭️ **5 skipped** integration tests (require live server)
- ❌ **0 failures**
- ⚠️ **0 warnings**
- 🎯 **Production ready**

**Recent Improvements (December 2025)**:
- ✅ Fixed all Pydantic V2 deprecation warnings
- ✅ Updated datetime handling to use `datetime.now(UTC)` (Python 3.13+ compatible)
- ✅ Resolved scoring function return value unpacking (2→3 values)
- ✅ Fixed hierarchical grouping score extraction bug (`breakdown.get("final_score")`)
- ✅ Updated PA scoring expectations for new linear formula
- ✅ Enhanced test resilience for scoring formula changes
- ✅ Implemented parallel ISN processing with `asyncio.gather()` (3-5x performance improvement)
- ✅ Increased cache duration for `/top-product/with-pa-trends` from 60s to 300s
- ✅ Added PA-adjusted measurements with specialized scoring logic
- ✅ Fixed Score Breakdown type definitions for PA vs standard measurements

To include DUT-router integration tests (requires intranet access and live server on port 8002):
```bash
pytest -m integration  # Run only integration tests
pytest -m "not integration"  # Skip integration tests (default behavior)
```

---

## Project Structure

```
backend_fastapi/
├── .github/workflows/ci.yml          # CI pipeline
├── data/                             # sample MC2/DVT data, criteria templates
├── docs/                             # project documentation
├── scripts/                          # CLI utilities (lint, init, debug)
├── src/app/
│   ├── main.py                       # FastAPI application entrypoint
│   ├── dependencies/                 # dependency providers (settings, auth)
│   ├── external_services/            # DUT API client wrapper
│   ├── models/                       # SQLAlchemy models (User, RBAC)
│   ├── routers/                      # API routers (auth, compare, multi-dut, etc.)
│   ├── schemas/                      # Pydantic response/request schemas
│   ├── services/                     # token / redis utilities
│   └── utils/                        # compare/multi-DUT processors, spec loaders
├── tests/                            # pytest suites (HTTP + unit tests)
├── Makefile                          # developer shortcuts
├── pyproject.toml                    # project metadata / dependencies
└── uv.lock                           # resolved dependency versions
```

---

## Common Commands

| Command          | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `make install`   | Install project in editable mode (`uv pip install -e .`) |
| `make dev`       | Hot-reload dev server on <http://127.0.0.1:8001>         |
| `make start`     | Uvicorn without reload (staging/production)              |
| `make format`    | Ruff format (`uv run ruff format`)                       |
| `make lint`      | Ruff lint (`uv run ruff check`)                          |
| `make fix`       | Ruff lint (`uv run ruff check --fix`)                    |
| `make test`      | Run pytest (`uv run pytest -v`)                          |
| `uv run uv lock` | Refresh `uv.lock` and `requirements.txt`                 |

---

## Tech Stack & Updating Dependencies

- **Python** >=3.13
- **FastAPI** + **Starlette**
- **SQLAlchemy** (PostgreSQL via `psycopg`)
- **Pydantic** / `pydantic-settings`
- **Redis** via `redis-py` (DUT tokens)
- **Pandas**, **OpenPyXL** (data processing / Excel)
- **pwdlib** (Argon2 via `argon2-cffi` for password hashing)
- **Uvicorn** for ASGI serving
- **Ruff** for lint/format
- **Pytest** for testing

### Upgrade workflow

1. Update a dependency (example: FastAPI)
   ```bash
   uv pip install --upgrade fastapi
   ```
2. Re-lock dependencies
   ```bash
   uv run uv lock          # regenerates uv.lock and requirements.txt
   ```
3. Run formatting/lint/tests before committing.
   ```bash
   make format && make lint && make test
   ```

---

## Database & Migration Notes

The project uses **Alembic** for database migrations and SQLAlchemy for ORM.

### Initial Setup

Run database migrations to create all tables:

```bash
make upgrade
# or manually:
uv run alembic upgrade head
```

### Creating New Migrations

When you modify models, create a new migration:

```bash
make migrate msg="add new column to users table"
# or manually:
uv run alembic revision --autogenerate -m "add new column to users table"
```

### Migration Commands

| Command | Description |
|---------|-------------|
| `make upgrade` | Apply all pending migrations |
| `make downgrade` | Rollback the last migration |
| `make migrate-history` | View migration history |
| `make migrate-current` | Show current migration version |
| `uv run alembic downgrade <revision>` | Rollback to specific revision |
| `uv run alembic history --verbose` | Inspect the revision graph with timestamps |
| `uv run alembic stamp head` | Mark the database as up-to-date without running migrations |

#### Alembic workflow tips
1. **Generate:** run `uv run alembic revision --autogenerate -m "short message"` and inspect the generated file under `alembic/versions/…`. Fine‑tune it before proceeding.
2. **Upgrade locally:** apply the revision with `uv run alembic upgrade head` and verify the schema/data changes.
3. **Rollback if needed:** use `uv run alembic downgrade -1` (or a specific revision) during testing.
4. **Promote:** once satisfied, commit both the model changes and the new `alembic/versions/*.py` file.

### Seeding Data

Sample script to create or update users:

```bash
uv run python scripts/create_user.py --username admin --password <your_password> --admin
```

For a full RBAC bootstrap (permissions, roles, and seeded users), run:

```bash
uv run python scripts/bootstrap_rbac.py
```

### RBAC quickstart
1. **Bootstrap canonical roles/permissions**
   ```bash
   uv run python scripts/bootstrap_rbac.py \
     --admin-password "<AdminPass123>" \
     --analyst-password "<AnalystPass123>" \
     --viewer-password "<ViewerPass123>"
   ```
   This creates the base `read`, `write`, `analyze`, `manage_users` permissions, three roles (`admin`, `analyst`, `viewer`), and optional seed users.

2. **Create additional users**
   ```bash
   uv run python scripts/create_user.py --username qa_user --password "<StrongPass!>" 
   uv run python scripts/create_user.py --username ops_admin --admin
   ```
   The script prints whether the user was created or updated and sets the `is_admin` flag when `--admin` is provided.

3. **Create custom roles or permissions via API (requires an admin token)**
   ```bash
   curl -X POST http://127.0.0.1:8001/api/rbac/roles \
     -H "Authorization: Bearer <ADMIN_ACCESS_TOKEN>" \
     -F "name=data_scientist" \
     -F "description=Access to comparison and analytics endpoints"
   ```
   Grant specific permissions with `/api/rbac/roles/{role_id}/grant`.

4. **Assign a role to a user**
   - Find the user ID (e.g., `SELECT id, username FROM app_user;` in PostgreSQL or a quick Python snippet).
   - Call the assignment endpoint:
     ```bash
     curl -X POST http://127.0.0.1:8001/api/rbac/users/7/assign-role \
       -H "Authorization: Bearer <ADMIN_ACCESS_TOKEN>" \
       -F "role_id=3"
     ```
   Use `/api/rbac/users/{user_id}/remove-role` to detach roles when needed.

---

## 📚 API Endpoint Reference

Base URL: `http://127.0.0.1:8001`

All responses are JSON unless otherwise noted. Most endpoints require authentication via Bearer token.

---

## 1. 🏠 Root & Documentation Endpoints

### `GET /`

**Description:** Health check endpoint that returns the API status.

**Authentication:** None required

**Response:**
```json
{
  "message": "DUT Management API"
}
```

**Example Request:**
```bash
curl http://127.0.0.1:8001/
```

---

### `GET /swagger`

**Description:** Interactive Swagger UI for API exploration and testing.

**Authentication:** None required (but individual endpoints may require auth)

**Access:** Navigate to `http://127.0.0.1:8001/swagger` in your browser

---

### `GET /redoc`

**Description:** ReDoc documentation UI with clean, readable API documentation.

**Authentication:** None required

**Access:** Navigate to `http://127.0.0.1:8001/redoc` in your browser

---

## 2. 🔐 Authentication Endpoints (`/api/auth`)

### `POST /api/auth/login`

**Description:** Authenticate with local database credentials and receive JWT access and refresh tokens.

**Authentication:** None required

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | Yes | Username for authentication | `admin` |
| `password` | string | Yes | User password | `admin123` |

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Response Fields:**
- `access_token`: JWT token for authenticating API requests (expires in 24 hours)
- `refresh_token`: JWT token for obtaining new access tokens (expires in 7 days)
- `token_type`: Always "bearer"

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/auth/login \
  -F "username=admin" \
  -F "password=admin123"
```

**Status Codes:**
- `200 OK`: Successful authentication
- `401 Unauthorized`: Invalid credentials

---

### `POST /api/auth/external-login`

**Description:** Authenticate against the external DUT API, cache upstream tokens, and issue local JWT tokens.

**Authentication:** None required

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | Yes | DUT API username | `oa_username` |
| `password` | string | Yes | DUT API password | `oa_password` |

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/auth/external-login \
  -F "username=dut_user" \
  -F "password=dut_password"
```

**Notes:**
- Creates/updates local user account if doesn't exist
- Stores upstream DUT tokens in Redis (or in-memory cache)
- Returns local JWT tokens for this API
- Requires connectivity to DUT API server

**Status Codes:**
- `200 OK`: Successful authentication
- `401 Unauthorized`: External authentication failed
- `500 Internal Server Error`: DUT API connection error

---

### `GET /api/auth/me`

**Description:** Get current authenticated user's profile information.

**Authentication:** Bearer token required

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "username": "admin",
  "is_admin": true,
  "roles": ["admin", "analyst"]
}
```

**Response Fields:**
- `username`: User's username
- `is_admin`: Boolean indicating admin privileges
- `roles`: Array of role names assigned to the user

**Example Request:**
```bash
curl http://127.0.0.1:8001/api/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Status Codes:**
- `200 OK`: Successfully retrieved user info
- `401 Unauthorized`: Invalid or expired token

---

### `POST /api/auth/token/refresh`

**Description:** Exchange a refresh token for new access and refresh tokens.

**Authentication:** None required (uses refresh token in request body)

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `refresh_token` | string | Yes | Valid JWT refresh token | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/auth/token/refresh \
  -F "refresh_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Notes:**
- Both tokens are rotated (new tokens issued)
- Old tokens become invalid after refresh
- Token versioning enforces revocation

**Status Codes:**
- `200 OK`: Tokens refreshed successfully
- `401 Unauthorized`: Invalid or expired refresh token

---

### `POST /api/auth/users/create`

**Description:** Create a new user or update an existing user. Requires admin privileges.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | Yes | Username for new user | `analyst_01` |
| `password` | string | Yes | User password | `SecurePass123` |
| `is_admin` | boolean | No | Grant admin privileges (default: false) | `true` |

**Response:**
```json
{
  "username": "analyst_01",
  "is_admin": false
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/auth/users/create \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "username=analyst_01" \
  -F "password=SecurePass123" \
  -F "is_admin=false"
```

**Notes:**
- Passwords are hashed using Argon2
- If user exists, updates their information
- Admin flag can be toggled via this endpoint

**Status Codes:**
- `200 OK`: User created/updated successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user

---

### `POST /api/auth/users/change-password`

**Description:** Change a user's password. Users can change their own password, admins can change any user's password.

**Authentication:** Bearer token required

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | Yes | Username to change password for | `analyst_01` |
| `new_password` | string | Yes | New password | `UpdatedPass456` |

**Response:**
```json
{
  "username": "analyst_01",
  "changed": true,
  "revoked_tokens": true
}
```

**Response Fields:**
- `username`: Username that was updated
- `changed`: Boolean confirming password change
- `revoked_tokens`: Boolean confirming all tokens were invalidated

**Example Request:**
```bash
# Change own password
curl -X POST http://127.0.0.1:8001/api/auth/users/change-password \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "username=analyst_01" \
  -F "new_password=UpdatedPass456"

# Admin changing another user's password
curl -X POST http://127.0.0.1:8001/api/auth/users/change-password \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "username=other_user" \
  -F "new_password=NewPass789"
```

**Notes:**
- All existing tokens for the user are revoked (token version incremented)
- User must re-authenticate after password change

**Status Codes:**
- `200 OK`: Password changed successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not authorized (not admin and not changing own password)
- `404 Not Found`: User not found

---

### `DELETE /api/auth/users/{username}`

**Description:** Permanently delete a user account. Requires admin privileges.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `username` | string | Yes | Username to delete | `analyst_01` |

**Response:**
```json
{
  "deleted": "analyst_01"
}
```

**Example Request:**
```bash
curl -X DELETE http://127.0.0.1:8001/api/auth/users/analyst_01 \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

**Status Codes:**
- `200 OK`: User deleted successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: User not found

---

### `POST /api/auth/users/revoke`

**Description:** Revoke all access and refresh tokens for a user by incrementing their token version. Requires admin privileges.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | Yes | Username to revoke tokens for | `analyst_01` |

**Response:**
```json
{
  "revoked": "analyst_01",
  "token_version": 2
}
```

**Response Fields:**
- `revoked`: Username whose tokens were revoked
- `token_version`: New token version number

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/auth/users/revoke \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "username=analyst_01"
```

**Notes:**
- All existing tokens become invalid immediately
- User must re-authenticate to get new tokens
- Token version is checked on every request

**Status Codes:**
- `200 OK`: Tokens revoked successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: User not found

---

### `POST /api/auth/external-token`

**Description:** Retrieve an external DUT API access token. Requires admin privileges. Used for server-to-server communication.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `username` | string | No | DUT API username (uses service account if omitted) | `service_account` |
| `password` | string | No | DUT API password (uses service account if omitted) | `ServicePass789` |

**Response:**
```json
{
  "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response Fields:**
- `access`: External DUT API access token

**Example Request:**
```bash
# Using service account credentials from environment
curl -X POST http://127.0.0.1:8001/api/auth/external-token \
  -H "Authorization: Bearer <ADMIN_TOKEN>"

# Using custom credentials
curl -X POST http://127.0.0.1:8001/api/auth/external-token \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "username=custom_user" \
  -F "password=custom_pass"
```

**Status Codes:**
- `200 OK`: Token retrieved successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `500 Internal Server Error`: External API authentication failed

---

## 3. 👥 RBAC Endpoints (`/api/rbac`)

Role-Based Access Control (RBAC) endpoints for managing roles, permissions, and their assignments. All endpoints require admin privileges.

### `POST /api/rbac/roles`

**Description:** Create a new role with optional description.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `name` | string | Yes | Unique role name | `data_analyst` |
| `description` | string | No | Role description | `Can review dashboards and reports` |

**Response:**
```json
{
  "id": 3,
  "name": "data_analyst",
  "description": "Can review dashboards and reports"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/roles \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "name=data_analyst" \
  -F "description=Can review dashboards and reports"
```

**Status Codes:**
- `200 OK`: Role created successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `409 Conflict`: Role already exists

---

### `GET /api/rbac/roles`

**Description:** List all roles with their assigned permissions.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Response:**
```json
{
  "roles": [
    {
      "id": 1,
      "name": "admin",
      "permissions": ["read", "write", "delete", "manage_users"]
    },
    {
      "id": 2,
      "name": "analyst",
      "permissions": ["read", "write"]
    }
  ]
}
```

**Example Request:**
```bash
curl http://127.0.0.1:8001/api/rbac/roles \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

**Status Codes:**
- `200 OK`: Roles retrieved successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user

---

### `DELETE /api/rbac/roles/{role_id}`

**Description:** Delete an existing role by ID.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `role_id` | integer | Yes | Role ID to delete | `3` |

**Response:**
```json
{
  "deleted": 3
}
```

**Example Request:**
```bash
curl -X DELETE http://127.0.0.1:8001/api/rbac/roles/3 \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

**Status Codes:**
- `200 OK`: Role deleted successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: Role not found

---

### `POST /api/rbac/permissions`

**Description:** Create a new permission with optional description.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `name` | string | Yes | Unique permission name | `read_reports` |
| `description` | string | No | Permission description | `Allows viewing report data` |

**Response:**
```json
{
  "id": 5,
  "name": "read_reports",
  "description": "Allows viewing report data"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/permissions \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "name=read_reports" \
  -F "description=Allows viewing report data"
```

**Status Codes:**
- `200 OK`: Permission created successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `409 Conflict`: Permission already exists

---

### `GET /api/rbac/permissions`

**Description:** List all permissions currently defined in the system.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Response:**
```json
{
  "permissions": [
    {
      "id": 1,
      "name": "read"
    },
    {
      "id": 2,
      "name": "write"
    },
    {
      "id": 3,
      "name": "delete"
    }
  ]
}
```

**Example Request:**
```bash
curl http://127.0.0.1:8001/api/rbac/permissions \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

**Status Codes:**
- `200 OK`: Permissions retrieved successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user

---

### `DELETE /api/rbac/permissions/{perm_id}`

**Description:** Delete a permission by ID.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `perm_id` | integer | Yes | Permission ID to delete | `5` |

**Response:**
```json
{
  "deleted": 5
}
```

**Example Request:**
```bash
curl -X DELETE http://127.0.0.1:8001/api/rbac/permissions/5 \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

**Status Codes:**
- `200 OK`: Permission deleted successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: Permission not found

---

### `POST /api/rbac/roles/{role_id}/grant`

**Description:** Grant a permission to a role (attach permission to role).

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `role_id` | integer | Yes | Role ID to grant permission to | `2` |

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `perm_id` | integer | Yes | Permission ID to grant | `5` |

**Response:**
```json
{
  "role": "analyst",
  "granted": "read_reports"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/roles/2/grant \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "perm_id=5"
```

**Status Codes:**
- `200 OK`: Permission granted successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: Role or permission not found

---

### `POST /api/rbac/roles/{role_id}/revoke`

**Description:** Revoke a permission from a role (detach permission from role).

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `role_id` | integer | Yes | Role ID to revoke permission from | `2` |

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `perm_id` | integer | Yes | Permission ID to revoke | `5` |

**Response:**
```json
{
  "role": "analyst",
  "revoked": "read_reports"
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/roles/2/revoke \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "perm_id=5"
```

**Status Codes:**
- `200 OK`: Permission revoked successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: Role or permission not found

---

### `POST /api/rbac/users/{user_id}/assign-role`

**Description:** Assign a role to a user.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `user_id` | integer | Yes | User ID to assign role to | `5` |

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `role_id` | integer | Yes | Role ID to assign | `2` |

**Response:**
```json
{
  "user": "analyst_01",
  "role": "analyst",
  "assigned": true
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/users/5/assign-role \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "role_id=2"
```

**Status Codes:**
- `200 OK`: Role assigned successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: User or role not found

---

### `POST /api/rbac/users/{user_id}/remove-role`

**Description:** Remove a role from a user.

**Authentication:** Bearer token required (Admin only)

**Request Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `user_id` | integer | Yes | User ID to remove role from | `5` |

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `role_id` | integer | Yes | Role ID to remove | `2` |

**Response:**
```json
{
  "user": "analyst_01",
  "role": "analyst",
  "removed": true
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/rbac/users/5/remove-role \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -F "role_id=2"
```

**Status Codes:**
- `200 OK`: Role removed successfully
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an admin user
- `404 Not Found`: User or role not found

---

### `GET /api/rbac/demo/needs-read`

**Description:** Demo endpoint that requires 'read' permission. Used for testing permission-based access control.

**Authentication:** Bearer token required (requires 'read' permission)

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "ok": true,
  "message": "You have 'read' permission."
}
```

**Example Request:**
```bash
curl http://127.0.0.1:8001/api/rbac/demo/needs-read \
  -H "Authorization: Bearer <TOKEN_WITH_READ_PERMISSION>"
```

**Status Codes:**
- `200 OK`: User has required permission
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: User lacks 'read' permission

---

## 4. 📁 Upload & Parsing Endpoints (`/api/upload-preview`, `/api/parse`, `/api/cleanup-uploads`)

### `POST /api/upload-preview`

**Description:** Upload a CSV or Excel file and receive a preview with column names and sample rows (first 20 rows).

**Authentication:** None required

**Request Body (Multipart Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `file` | file | Yes | CSV or Excel file to upload | `sample.csv` |
| `has_header` | boolean | No | Whether first row contains headers (default: true) | `true` |
| `delimiter` | string | No | CSV delimiter (auto-detected if not provided) | `,` |
| `persist` | boolean | No | Override environment persistence setting | `false` |

**Response:**
```json
{
  "file_id": "abcd1234ef567890_sample.csv",
  "filename": "sample.csv",
  "columns": ["ISN", "Result", "Timestamp", "Device"],
  "preview": [
    {
      "ISN": "260884980003907",
      "Result": "PASS",
      "Timestamp": "2024-10-15 10:30:00",
      "Device": "Device_01"
    },
    {
      "ISN": "260884980003908",
      "Result": "FAIL",
      "Timestamp": "2024-10-15 10:35:00",
      "Device": "Device_02"
    }
  ]
}
```

**Response Fields:**
- `file_id`: Unique identifier for the uploaded file (use in subsequent requests)
- `filename`: Original filename
- `columns`: Array of column names
- `preview`: Array of row objects (max 20 rows)

**Example Request:**
```bash
# Upload CSV with headers
curl -X POST http://127.0.0.1:8001/api/upload-preview \
  -F "file=@data/sample_data/test.csv" \
  -F "has_header=true"

# Upload Excel file with custom delimiter
curl -X POST http://127.0.0.1:8001/api/upload-preview \
  -F "file=@data/sample_data/test.xlsx" \
  -F "delimiter=;" \
  -F "persist=true"
```

**Notes:**
- Supports CSV and Excel (.xls, .xlsx) files
- Auto-detects CSV delimiter if not specified
- File is stored in-memory or on disk based on `UPLOAD_PERSIST` env var
- Background cleanup worker removes expired uploads based on TTL

**Status Codes:**
- `200 OK`: File uploaded and previewed successfully
- `400 Bad Request`: Invalid file format or parsing error
- `500 Internal Server Error`: Failed to process file

---

### `POST /api/parse`

**Description:** Parse uploaded file and extract specific rows/columns, returning JSON data.

**Authentication:** None required

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `file_id` | string | Yes | File identifier from upload-preview | `abcd1234ef567890_sample.csv` |
| `mode` | string | Yes | Selection mode: `columns`, `rows`, or `both` | `columns` |
| `selected_columns` | string | No | JSON array of column names/indices to include | `["ISN", "Result"]` or `[0, 2]` |
| `selected_rows` | string | No | JSON array of row indices to include | `[0, 2, 4]` |
| `exclude_columns` | string | No | JSON array of column names/indices to exclude | `["Timestamp"]` |
| `exclude_rows` | string | No | JSON array of row indices to exclude | `[1, 3]` |

**Response:**
```json
{
  "columns": ["ISN", "Result"],
  "rows": [
    {
      "ISN": "260884980003907",
      "Result": "PASS"
    },
    {
      "ISN": "260884980003908",
      "Result": "FAIL"
    }
  ]
}
```

**Response Fields:**
- `columns`: Selected column names
- `rows`: Array of row objects with selected columns

**Example Requests:**
```bash
# Select specific columns
curl -X POST http://127.0.0.1:8001/api/parse \
  -F "file_id=abcd1234_sample.csv" \
  -F "mode=columns" \
  -F 'selected_columns=["ISN", "Result"]'

# Select specific rows
curl -X POST http://127.0.0.1:8001/api/parse \
  -F "file_id=abcd1234_sample.csv" \
  -F "mode=rows" \
  -F "selected_rows=[0, 1, 2]"

# Select columns and rows, exclude some
curl -X POST http://127.0.0.1:8001/api/parse \
  -F "file_id=abcd1234_sample.csv" \
  -F "mode=both" \
  -F 'selected_columns=["ISN", "Result", "Device"]' \
  -F "selected_rows=[0, 2, 4, 6]" \
  -F 'exclude_columns=["Device"]'
```

**Notes:**
- Columns can be specified by name or index (0-based)
- Rows are specified by index (0-based)
- Exclusions are applied after selections
- Missing values are filled with empty strings

**Status Codes:**
- `200 OK`: Data parsed successfully
- `400 Bad Request`: Invalid selection parameters or file not found
- `500 Internal Server Error`: Parsing error

---

### `POST /api/parse-download`

**Description:** Parse uploaded file and download selected data as CSV.

**Authentication:** None required

**Request Body (Form Data):**
Same parameters as `/api/parse` endpoint.

**Response:**
CSV file download with selected data.

**Response Headers:**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="parsed.csv"
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/parse-download \
  -F "file_id=abcd1234_sample.csv" \
  -F "mode=columns" \
  -F 'selected_columns=["ISN", "Result"]' \
  -o parsed_output.csv
```

**Status Codes:**
- `200 OK`: CSV downloaded successfully
- `400 Bad Request`: Invalid parameters or file not found

---

### `POST /api/cleanup-uploads`

**Description:** Manually trigger cleanup of expired uploaded files (both on-disk and in-memory).

**Authentication:** Admin key required (if `ASTPARSER_ADMIN_KEY` is set in environment)

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `admin_key` | string | Conditional | Admin key (required if ASTPARSER_ADMIN_KEY is set) | `supersecretkey` |
| `ttl` | integer | No | Time-to-live override in seconds | `3600` |

**Response:**
```json
{
  "removed": [
    "abcd1234_old_file.csv",
    "efgh5678_another_file.xlsx"
  ]
}
```

**Response Fields:**
- `removed`: Array of removed file identifiers

**Example Requests:**
```bash
# Cleanup with default TTL
curl -X POST http://127.0.0.1:8001/api/cleanup-uploads \
  -F "admin_key=supersecretkey"

# Cleanup with custom TTL (1 hour)
curl -X POST http://127.0.0.1:8001/api/cleanup-uploads \
  -F "admin_key=supersecretkey" \
  -F "ttl=3600"

# If ASTPARSER_ADMIN_KEY is not set (testing)
curl -X POST http://127.0.0.1:8001/api/cleanup-uploads
```

**Notes:**
- Background cleanup worker runs automatically
- This endpoint allows manual cleanup on-demand
- Default TTL from `UPLOAD_TTL_SECONDS` env var
- Admin key check is skipped in test environments

**Status Codes:**
- `200 OK`: Cleanup completed
- `403 Forbidden`: Invalid admin key

---

## 5. ⚖️ Comparison Endpoints (`/api/compare`)

### `POST /api/compare`

**Description:** Compare two uploaded files cell-by-cell with optional join on specific columns. Returns differences and matching rows.

**Authentication:** None required

**Request Body (Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `file_a` | string | Yes | First file identifier | `abc123_fileA.csv` |
| `file_b` | string | Yes | Second file identifier | `def456_fileB.csv` |
| `mode` | string | Yes | Comparison mode: `columns`, `rows`, or `both` | `both` |
| `a_selected_columns` | string | No | JSON array of columns to select from file A | `["ISN", "Result"]` |
| `a_selected_rows` | string | No | JSON array of row indices from file A | `[0, 1, 2]` |
| `a_exclude_columns` | string | No | JSON array of columns to exclude from file A | `["Timestamp"]` |
| `a_exclude_rows` | string | No | JSON array of row indices to exclude from file A | `[5, 6]` |
| `b_selected_columns` | string | No | JSON array of columns to select from file B | `["ISN", "Result"]` |
| `b_selected_rows` | string | No | JSON array of row indices from file B | `[0, 1, 2]` |
| `b_exclude_columns` | string | No | JSON array of columns to exclude from file B | `["Operator"]` |
| `b_exclude_rows` | string | No | JSON array of row indices to exclude from file B | `[3, 4]` |
| `a_join_on` | string | No | Column(s) to join on in file A | `["ISN"]` or `"ISN"` |
| `b_join_on` | string | No | Column(s) to join on in file B | `["ISN"]` or `"ISN"` |

**Response:**
```json
{
  "rows": [
    {
      "ISN": "260884980003907",
      "Result_A": "PASS",
      "Result_B": "PASS",
      "status": "match"
    },
    {
      "ISN": "260884980003908",
      "Result_A": "PASS",
      "Result_B": "FAIL",
      "status": "different",
      "differences": ["Result"]
    }
  ],
  "summary": {
    "total_rows": 10,
    "matching": 8,
    "different": 2,
    "only_in_a": 1,
    "only_in_b": 1
  }
}
```

**Response Fields:**
- `rows`: Array of comparison results
  - `status`: `match`, `different`, `only_in_a`, or `only_in_b`
  - `differences`: Array of column names with differences (if status is `different`)
- `summary`: Statistics about the comparison
  - `total_rows`: Total rows compared
  - `matching`: Rows with identical values
  - `different`: Rows with differing values
  - `only_in_a`: Rows only in first file
  - `only_in_b`: Rows only in second file

**Example Requests:**
```bash
# Simple column comparison
curl -X POST http://127.0.0.1:8001/api/compare \
  -F "file_a=abc123_test1.csv" \
  -F "file_b=def456_test2.csv" \
  -F "mode=columns" \
  -F 'a_selected_columns=["ISN", "Result"]' \
  -F 'b_selected_columns=["ISN", "Result"]'

# Comparison with join on ISN column
curl -X POST http://127.0.0.1:8001/api/compare \
  -F "file_a=abc123_golden.csv" \
  -F "file_b=def456_test.csv" \
  -F "mode=both" \
  -F 'a_selected_columns=["ISN", "Power", "Frequency"]' \
  -F 'b_selected_columns=["ISN", "Power", "Frequency"]' \
  -F 'a_join_on=["ISN"]' \
  -F 'b_join_on=["ISN"]'

# Row-based comparison with exclusions
curl -X POST http://127.0.0.1:8001/api/compare \
  -F "file_a=abc123_data.csv" \
  -F "file_b=def456_data.csv" \
  -F "mode=rows" \
  -F "a_selected_rows=[0,1,2,3,4]" \
  -F "b_selected_rows=[0,1,2,3,4]" \
  -F 'a_exclude_columns=["Timestamp"]' \
  -F 'b_exclude_columns=["Timestamp"]'
```

**Notes:**
- Join functionality matches rows based on specified columns
- Columns are matched by name or position
- Cell-by-cell comparison for differing values
- Handles missing columns gracefully

**Status Codes:**
- `200 OK`: Comparison completed successfully
- `400 Bad Request`: Invalid parameters or files not found
- `500 Internal Server Error`: Comparison error

---

### `POST /api/compare-download`

**Description:** Compare two files and download results as CSV.

**Authentication:** None required

**Request Body (Form Data):**
Same parameters as `/api/compare` endpoint.

**Response:**
CSV file download with comparison results.

**Response Headers:**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="compare.csv"
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/compare-download \
  -F "file_a=abc123_test1.csv" \
  -F "file_b=def456_test2.csv" \
  -F "mode=both" \
  -F 'a_selected_columns=["ISN", "Result"]' \
  -F 'b_selected_columns=["ISN", "Result"]' \
  -F 'a_join_on="ISN"' \
  -F 'b_join_on="ISN"' \
  -o comparison_results.csv
```

**Status Codes:**
- `200 OK`: CSV downloaded successfully
- `400 Bad Request`: Invalid parameters or files not found

---

## 6. 📊 Format Comparison Endpoint (`/api/compare-formats`)

### `POST /api/compare-formats`

**Description:** Compare MasterControl (MC2) and DVT (Design Verification Test) formatted test data with specification validation. Returns pass/fail status and differences.

**Authentication:** None required

**Request Body (Multipart Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `master_file` | file | Yes | MasterControl format test file | `MC2_data.csv` |
| `dvt_file` | file | Yes | DVT format test file | `DVT_Golden_SN.csv` |
| `threshold` | float | No | Comparison tolerance threshold | `1.0` |
| `margin_threshold` | float | No | Pass/fail margin threshold override | `0.5` |
| `spec_file` | file | No | JSON specification file (uses default if omitted) | `compare_format_config.json` |
| `freq_tol` | float | No | Frequency tolerance in MHz (default: 2.0) | `2.0` |
| `human` | boolean | No | Return human-readable CSV/XLSX instead of JSON | `true` |
| `return_xlsx` | boolean | No | Return XLSX format (requires human=true) | `true` |

**Spec Format:** JSON only. The file must match the structure used in `data/sample_data_config/compare_format_config.json` (per-standard arrays describing `frequency`, `bw`, `mod`, `tx_target_power`, the associated limits, and optional `tx_target_tolerance`). INI/criteria files are **not** accepted on this endpoint.

**Response (JSON mode):**
```json
{
  "rows": [
    {
      "antenna_dvt": 0,
      "metric": "POW",
      "freq": 2412,
      "standard": "11AX",
      "datarate": "MCS11",
      "bandwidth": "B160",
      "usl": 23.0,
      "lsl": 15.0,
      "mc2_value": 19.5,
      "mc2_spec_diff": "4.5 below USL",
      "mc2_result": "PASS",
      "dvt_value": 19.3,
      "dvt_spec_diff": "4.7 below USL",
      "dvt_result": "PASS",
      "mc2_dvt_diff": 0.2
    }
  ],
  "summary": {
    "pass": 45,
    "fail": 5,
    "total": 50
  }
}
```

**Response (CSV/XLSX mode):**
CSV or Excel file with columns:
- Antenna, Test Mode, Metric, Freq, Standard, DataRate, BW
- USL, LSL, MC2 Value, MC2 & Spec Diff, MC2 Result
- DVT Value, DVT & Spec Diff, DVT Result, MC2 & DVT Diff

**Response Fields:**
- `antenna_dvt`: Antenna number (0-based)
- `metric`: Test metric (POW, EVM, MASK, FREQ, etc.)
- `freq`: Frequency in MHz
- `standard`: Wireless standard (11AX, 11AC, etc.)
- `datarate`: Data rate (MCS11, MCS9, etc.)
- `bandwidth`: Channel bandwidth (B20, B40, B80, B160)
- `usl`: Upper specification limit
- `lsl`: Lower specification limit
- `mc2_value`: MasterControl measured value
- `mc2_spec_diff`: Difference from specification
- `mc2_result`: PASS or FAIL
- `dvt_value`: DVT measured value
- `dvt_spec_diff`: Difference from specification
- `dvt_result`: PASS or FAIL
- `mc2_dvt_diff`: Difference between MC2 and DVT values

**Example Requests:**
```bash
# JSON comparison
curl -X POST http://127.0.0.1:8001/api/compare-formats \
  -F "master_file=@data/sample_data/MC2_25G.csv" \
  -F "dvt_file=@data/sample_data/DVT_Golden_25G.csv" \
  -F "threshold=1.0"

# CSV output with custom spec
curl -X POST http://127.0.0.1:8001/api/compare-formats \
  -F "master_file=@data/sample_data/MC2_25G.csv" \
  -F "dvt_file=@data/sample_data/DVT_Golden_25G.csv" \
  -F "spec_file=@data/sample_data_config/custom_spec.json" \
  -F "threshold=0.5" \
  -F "freq_tol=1.5" \
  -F "human=true" \
  -o comparison.csv

# XLSX output with margin threshold
curl -X POST http://127.0.0.1:8001/api/compare-formats \
  -F "master_file=@data/sample_data/MC2_data.csv" \
  -F "dvt_file=@data/sample_data/DVT_data.csv" \
  -F "margin_threshold=1.0" \
  -F "human=true" \
  -F "return_xlsx=true" \
  -o Golden_Compare_Compiled.xlsx
```

**Notes:**
- Automatically matches test entries by frequency (within tolerance)
- Categorizes tests by mode: TX, RX, or Others
- Applies USL/LSL limits from spec file
- Calculates pass/fail based on margin threshold
- Includes provenance metadata (timestamp, thresholds) in output
- Falls back to `spec_config.json` if no spec file provided
- Output filename includes UTC+7 timestamp

**Status Codes:**
- `200 OK`: Comparison completed successfully
- `400 Bad Request`: File parsing error or invalid format
- `500 Internal Server Error`: Comparison failed

---

## 7. 🔄 DVT to MC2 Conversion (`/api/convert-dvt-to-mc2`)

### `POST /api/convert-dvt-to-mc2`

**Description:** Convert DVT (Design Verification Test) format files to MasterControl (MC2) format with configurable test specifications.

**Authentication:** None required

**Request Body (Multipart Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `dvt_file` | file | Yes | DVT format CSV file to convert | `DVT_Golden_SN_9403.csv` |
| `spec_json` | file | Yes | JSON specification file defining test parameters | `spec_config.json` |
| `output_format` | string | No | Output format: `csv` or `xlsx` (default: csv) | `xlsx` |

**Spec JSON Structure:**
```json
{
  "test_mode_tests": {
    "TX": [
      {
        "Metric": "POW",
        "Standard": "11AX",
        "BW": ["B20", "B40", "B80", "B160"],
        "DataRate": ["MCS0", "MCS11"],
        "Freq": [2412, 2437, 2462, 5180, 5500, 5745, 5955, 6115, 6435, 6915],
        "USL": 23.0,
        "LSL": 15.0
      }
    ],
    "RX": [...],
    "Others": [...]
  }
}
```

**Response (CSV mode):**
CSV file with MasterControl format columns:
- Antenna, Test Mode, Metric, Freq, Standard, DataRate, BW
- Value, USL, LSL, Result, Spec Diff

**Response (XLSX mode):**
Excel file with formatted columns and conditional highlighting.

**Response Headers:**
```
Content-Type: text/csv  # or application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename="MC2_converted_<timestamp>.csv"
```

**Example Requests:**
```bash
# Convert to CSV
curl -X POST http://127.0.0.1:8001/api/convert-dvt-to-mc2 \
  -F "dvt_file=@data/sample_data/DVT_Golden_25G.csv" \
  -F "spec_json=@data/sample_data_config/spec_config.json" \
  -F "output_format=csv" \
  -o MC2_converted.csv

# Convert to XLSX
curl -X POST http://127.0.0.1:8001/api/convert-dvt-to-mc2 \
  -F "dvt_file=@data/sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv" \
  -F "spec_json=@data/sample_data_config/spec_config.json" \
  -F "output_format=xlsx" \
  -o MC2_converted.xlsx
```

**Notes:**
- Extracts test results from DVT CSV format
- Maps to MasterControl columns using spec file
- Generates USL/LSL based on test specifications
- Calculates pass/fail and spec differences
- Supports multiple test modes (TX, RX, Others)
- Output filename includes UTC+7 timestamp

**Status Codes:**
- `200 OK`: Conversion completed successfully
- `400 Bad Request`: Invalid DVT file or spec JSON format
- `500 Internal Server Error`: Conversion failed

---

## 8. 📈 Multi-DUT Analysis (`/api/analyze-multi-dut`)

### `POST /api/analyze-multi-dut`

**Description:** Analyze a multi-DUT MC2 export with configurable specs and produce an annotated XLSX workbook.

**Authentication:** None

**Request Body (Multipart Form Data):**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `mc2_file` | file | Yes | MC2 CSV/XLSX multi-DUT report | `2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv` |
| `spec_file` | file | Yes | Spec definition (`*.json` format or DUT-style criteria `*.ini`) | `multi-dut_all_specs.json` |

**Response:**
Excel file with multiple sheets:
1. **Summary**: Overall pass/fail statistics per DUT
2. **TX Tests**: Transmit test results with criteria
3. **RX Tests**: Receive test results with criteria
4. **Others**: Miscellaneous tests (frequency, etc.)
5. **Failures**: Aggregated list of all failures

**Response Headers:**
```
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Content-Disposition: attachment; filename="MultiDUT_Analysis_<timestamp>.xlsx"
```

**Excel Output Structure:**

**Summary Sheet:**
| DUT Serial | Total Tests | Passed | Failed | Pass Rate % |
|------------|-------------|--------|--------|-------------|
| 260884980003907 | 150 | 148 | 2 | 98.67% |
| 260884980003908 | 150 | 145 | 5 | 96.67% |

**Test Results Sheets:**
| DUT Serial | Metric | Standard | DataRate | BW | Freq | Value | USL | LSL | Result | Margin |
|------------|--------|----------|----------|----|----|-------|-----|-----|--------|--------|
| 260884980003907 | POW | 11AX | MCS11 | B20 | 2412 | 19.5 | 22.0 | 18.0 | PASS | 2.5 |

**Example Requests:**
```bash
# Analyze with JSON spec
curl -X POST http://127.0.0.1:8001/api/analyze-multi-dut \
  -F "mc2_file=@data/sample_data/2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv" \
  -F "spec_file=@data/sample_data_config/multi-dut_all_specs.json" \
  -o MultiDUT_Analysis_json.xlsx

# Analyze with DUT-style INI spec
curl -X POST http://127.0.0.1:8001/api/analyze-multi-dut \
  -F "mc2_file=@data/sample_data/2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv" \
  -F "spec_file=@reference/conf_local_spec.ini" \
  -o MultiDUT_Analysis_ini.xlsx
```

**Notes:**
- Accepts a single MC2 CSV/XLSX export containing multiple DUT measurements
- `spec_file` supports the legacy JSON structure *or* DUT criteria INI with regex rules
- Output workbook includes summary sheets plus value/non-value data with computed margins
- Automatically timestamps the filename when none is provided in the request

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid file/spec format or parsing error
- `500 Internal Server Error`: Analysis processing error

---

## 9. 🌐 DUT Management & External API (`/api/dut/*`)

### Overview
Integration with upstream Device Under Test (DUT) management system. Provides metadata, test records, history, and analytics for devices tested in production.

**Authentication:** All DUT endpoints require Bearer token (uses cached DUT credentials)

**Request Headers:**
```
Authorization: Bearer <ACCESS_TOKEN>
```

**Note:** Ensure you log in via `/api/auth/external-login` first so the DUT access token is cached.

---

### 9.1 Metadata Endpoints

#### `GET /api/dut/sites`

**Description:** List all manufacturing sites.

**Response:**
```json
[
  {
    "id": 1,
    "name": "Site_Bangkok",
    "location": "Bangkok, Thailand"
  },
  {
    "id": 2,
    "name": "Site_Shenzhen",
    "location": "Shenzhen, China"
  }
]
```

**Example Request:**
```bash
curl -X GET http://127.0.0.1:8001/api/dut/sites \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Sites retrieved successfully
- `401 Unauthorized`: Missing/invalid token
- `503 Service Unavailable`: Upstream DUT API unavailable

---

#### `GET /api/dut/sites/{site_id}/models`

**Description:** List all device models for a specific site. Accepts site ID or name.

**Path Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `site_id` | int/string | Site ID or name | `1` or `Site_Bangkok` |

**Response:**
```json
[
  {
    "id": 101,
    "name": "RTL8852BE_WiFi7",
    "site_id": 1,
    "site_name": "Site_Bangkok"
  }
]
```

**Example Requests:**
```bash
# By site ID
curl -X GET http://127.0.0.1:8001/api/dut/sites/1/models \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# By site name
curl -X GET http://127.0.0.1:8001/api/dut/sites/Site_Bangkok/models \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Models retrieved successfully
- `404 Not Found`: Site not found
- `401 Unauthorized`: Missing/invalid token

---

#### `GET /api/dut/models/{model_id}/stations`

**Description:** List all test stations for a specific device model. Accepts model ID or name.

**Path Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model_id` | int/string | Model ID or name | `101` or `RTL8852BE_WiFi7` |

**Response:**
```json
[
  {
    "id": 1001,
    "name": "Station_TX_Power",
    "model_id": 101,
    "model_name": "RTL8852BE_WiFi7",
    "test_type": "TX"
  }
]
```

**Example Requests:**
```bash
# By model ID
curl -X GET http://127.0.0.1:8001/api/dut/models/101/stations \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# By model name
curl -X GET "http://127.0.0.1:8001/api/dut/models/RTL8852BE_WiFi7/stations" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Stations retrieved successfully
- `404 Not Found`: Model not found
- `401 Unauthorized`: Missing/invalid token

---

### 9.2 Test Record Endpoints

#### `GET /api/dut/records/{station_id}/{dut_id}`

**Description:** Get detailed test records for a device at a specific station. Accepts station/DUT IDs or names.

> ⚡ **Performance Note** — When `site_identifier` / `model_identifier` are omitted the API now streams the upstream DUT payload directly so the `record` array preserves the provider’s ordering (`test_date`, `station`, `device`, `isn`, `error_item`, …) without extra catalog lookups. Supplying those filters still enforces validation (ensuring the station belongs to the supplied site/model) and enriches the response with the authoritative metadata.

**Path Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `station_id` | int/string | Station ID or name | `1001` or `Station_TX_Power` |
| `dut_id` | int/string | DUT ID or serial number | `50001` or `260884980003907` |

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `start_date` | string | No | Start date (ISO 8601) | `2024-10-15T00:00:00Z` |
| `end_date` | string | No | End date (ISO 8601) | `2024-10-20T23:59:59Z` |
| `limit` | int | No | Maximum results (default: 100) | `50` |

**Response:**
```json
{
  "station": {
    "id": 1001,
    "name": "Station_TX_Power"
  },
  "dut": {
    "id": 50001,
    "serial": "260884980003907"
  },
  "records": [
    {
      "id": 900001,
      "timestamp": "2024-10-20T15:45:00Z",
      "test_name": "POW_11AX_MCS11_B20_2412",
      "metric": "POW",
      "value": 19.5,
      "usl": 22.0,
      "lsl": 18.0,
      "result": "PASS",
      "operator": "OP_001"
    }
  ]
}
```

**Example Requests:**
```bash
# By IDs
curl -X GET http://127.0.0.1:8001/api/dut/records/1001/50001 \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# By names (URL-encoded)
curl -X GET "http://127.0.0.1:8001/api/dut/records/BP3%20Download/260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

# Numeric IDs
curl -X GET http://127.0.0.1:8001/api/dut/records/142/9410441 \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Records retrieved successfully
- `404 Not Found`: Station or DUT not found
- `401 Unauthorized`: Missing/invalid token

#### `POST /api/dut/test-items/latest/batch`

**Description:** Return the latest set of value-based measurements plus non-value and non-value-bin checks for multiple stations on a DUT.

**Highlights:**
- Accepts `{ "dut_isn": "...", "station_identifiers": ["Wireless_Test_6G", 142] }` and optional `site_identifier` / `model_identifier` hints.
- Each station block now returns the buckets in this order:

```json
{
  "station_id": 145,
  "station_name": "Wireless_Test_6G",
  "station_dut_id": 9473227,
  "station_dut_isn": "DM2520270073965",
  "value_test_items": [
    { "name": "TX1_FIXTURE_OR_DUT_PROBLEM_POW_2405_ZIGBEE", "usl": 23.0, "lsl": 15.0 }
  ],
  "nonvalue_test_items": [
    { "name": "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80", "usl": null, "lsl": null }
  ],
  "nonvalue_bin_test_items": [
    { "name": "BT_RX_VERIFY_PER", "usl": null, "lsl": null }
  ],
  "error": null
}
```

- The `status` field has been removed from every bucket to keep payloads lean; PASS/FAIL style data now surfaces in `nonvalue_test_items` only.
- Value items are derived from `GET /api/dut/records/{station_id}/{dut_id}` (with a fallback to the historical endpoint when the DUT service lacks a native “latest” view), so the field will no longer be empty for stations that skip the “latest” API.

---

### 9.3 History Endpoints

#### `GET /api/dut/history/isns`

**Description:** Get all ISNs (Internal Serial Numbers) tested within a time range.

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string | Yes | Device ISN to query variants for | `260884980003907` |

**Response:**
```json
{
  "isns": [
    "260884980003907",
    "260884980003908",
    "260884980003909"
  ],
  "count": 3
}
```

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8001/api/dut/history/isns?dut_isn=260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: ISNs retrieved successfully
- `400 Bad Request`: Missing ISN parameter
- `401 Unauthorized`: Missing/invalid token

---

#### `GET /api/dut/history/identifiers`

**Description:** List DUT IDs observed for the provided ISN (per station).

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string | Yes | Device ISN | `260884980003907` |

**Response:**
```json
{
  "identifiers": [
    {
      "station_name": "Station_TX_Power",
      "dut_id": 50001
    },
    {
      "station_name": "Station_RX_Sensitivity",
      "dut_id": 50002
    }
  ]
}
```

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8001/api/dut/history/identifiers?dut_isn=260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Identifiers retrieved successfully
- `400 Bad Request`: Missing ISN parameter
- `401 Unauthorized`: Missing/invalid token

---

#### `GET /api/dut/history/progression`

**Description:** Show which stations have recorded runs for the DUT (tested flag).

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string | Yes | Device ISN | `260884980003907` |

**Response:**
```json
{
  "isn": "260884980003907",
  "progression": [
    {
      "station_name": "Station_TX_Power",
      "timestamp": "2024-10-20T10:30:00Z",
      "tested": true
    },
    {
      "station_name": "Station_RX_Sensitivity",
      "timestamp": "2024-10-20T11:15:00Z",
      "tested": true
    }
  ]
}
```

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8001/api/dut/history/progression?dut_isn=260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Progression retrieved successfully
- `400 Bad Request`: Missing ISN parameter
- `404 Not Found`: ISN not found
- `401 Unauthorized`: Missing/invalid token

---

#### `GET /api/dut/history/results`

**Description:** Count passes/fails per station with per-run status.

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string | Yes | Device ISN | `260884980003907` |

**Response:**
```json
{
  "isn": "260884980003907",
  "total_tests": 150,
  "passed": 148,
  "failed": 2,
  "results": [
    {
      "station_name": "Station_TX_Power",
      "passes": 75,
      "fails": 0
    },
    {
      "station_name": "Station_RX_Sensitivity",
      "passes": 73,
      "fails": 2
    }
  ]
}
```

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8001/api/dut/history/results?dut_isn=260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Results retrieved successfully
- `400 Bad Request`: Missing ISN parameter
- `404 Not Found`: ISN not found
- `401 Unauthorized`: Missing/invalid token

---

### 9.4 Summary & Analytics

#### `POST /api/dut/top-product`

**Description:** Batch analysis of multiple DUT ISNs to identify top-performing products across selected stations. Returns ranked results with detailed scoring for each ISN.

**Authentication:** Bearer token required

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string[] | Yes | List of DUT ISNs to analyze (multiple values) | `dut_isn=ISN1&dut_isn=ISN2` |
| `station` | string[] | No | Filter by station IDs or names | `station=148&station=Wireless_Test_6G` |
| `site_identifier` | string | No | Site ID or name for validation | `PTB` or `2` |
| `model_identifier` | string | No | Model ID or name for validation | `HH5K` or `44` |
| `device_identifier` | string[] | No | Device IDs or names to filter | `device_identifier=614660` |
| `test_item_filter` | string[] | No | Regex patterns to include specific measurements | `test_item_filter=WiFi_TX_POW.*` |
| `exclude_test_item_filter` | string[] | No | Regex patterns to exclude measurements | `exclude_test_item_filter=.*OLD.*` |

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `criteria_file` | file | No | INI file with scoring criteria (uses defaults if omitted) |

**Response:**
```json
{
  "results": [
    {
      "dut_isn": "DM2520270073965",
      "test_result": [
        {
          "station_id": 145,
          "station_name": "Wireless_Test_6G",
          "data": [
            {
              "test_item": "WiFi_TX1_POW_6185_11AX_MCS11_B160",
              "usl": 25.0,
              "lsl": 17.0,
              "target_used": 21.0,
              "actual": 20.5,
              "score": 9.5,
              "pass": true
            }
          ],
          "overall_score": 9.15
        }
      ]
    }
  ],
  "errors": []
}
```

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8001/api/dut/top-product?dut_isn=ISN1&dut_isn=ISN2&station=148" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "criteria_file=@criteria.ini"
```

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid parameters or missing required ISNs
- `401 Unauthorized`: Missing/invalid token

---

#### `POST /api/dut/top-product/with-pa-trends`

**Description:** Extended version of `/top-product` that **includes PA trend measurements** with adjusted power calculations. Processes PA SROM OLD/NEW pairs to calculate power adjustments alongside standard measurements.

**⚡ Performance:** Uses parallel ISN processing with `asyncio.gather()` for 3-5x faster execution compared to sequential processing. Cache duration: 300 seconds.

**Use Cases:**
- Analyze products with PA (Power Amplifier) calibration data
- Compare standard measurements + PA adjustments in single request
- Production quality verification with power calibration trends

**Note:** Use `/top-product` (without PA trends) for faster responses when PA data is not needed.

**Authentication:** Bearer token required

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string[] | Yes | List of DUT ISNs to analyze | `dut_isn=ISN1&dut_isn=ISN2` |
| `station` | string[] | No | Filter by station IDs or names | `station=148` |
| `site_identifier` | string | No | Site ID or name | `PTB` or `2` |
| `model_identifier` | string | No | Model ID or name | `HH5K` or `44` |
| `device_identifier` | string[] | No | Device IDs or names | `device_identifier=614660` |
| `test_item_filter` | string[] | No | Include patterns | `test_item_filter=WiFi_PA.*` |
| `exclude_test_item_filter` | string[] | No | Exclude patterns | `exclude_test_item_filter=.*OLD$` |

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `criteria_file` | file | No | INI scoring criteria file |

**PA Trend Integration:**
- Automatically fetches PA trend data for test items matching `PA{1-4}_SROM_OLD` and `PA{1-4}_SROM_NEW` patterns
- Pairs OLD/NEW items and calculates adjusted power: `(NEW - OLD) / 256`
- Creates synthetic `PA{1-4}_ADJUSTED_POW` test items with adjusted values
- Applies specialized scoring for PA-adjusted measurements

**Response Format:**
```json
{
  "results": [
    {
      "dut_isn": "DM2520270073965",
      "test_result": [
        {
          "station_id": 145,
          "station_name": "Wireless_Test_6G",
          "data": [
            {
              "test_item": "WiFi_TX1_POW_6185_11AX_MCS11_B160",
              "usl": 25.0,
              "lsl": 17.0,
              "target_used": 21.0,
              "actual": 20.5,
              "score": 9.5,
              "pass": true
            },
            {
              "test_item": "WiFi_PA1_ADJUSTED_POW_5985_11AX_MCS9_B80",
              "comparison": "within",
              "threshold": 0.5,
              "target_used": null,
              "current_value": 0.37,
              "trend_mean": 0.32,
              "deviation_from_mean": 0.05,
              "abs_deviation": 0.05,
              "interpretation": "Slightly above trend",
              "score": 9.2,
              "pass": true
            }
          ],
          "overall_score": 9.35
        }
      ]
    }
  ],
  "errors": []
}
```

**PA Measurement Fields:**
- `comparison`: Relationship to trend ("within", "above", "below")
- `threshold`: Acceptable deviation threshold (default: 0.5)
- `current_value`: Calculated adjusted power for this DUT
- `trend_mean`: Historical mean adjusted power from trend data
- `deviation_from_mean`: Difference from historical mean
- `abs_deviation`: Absolute value of deviation
- `interpretation`: Human-readable status message
- `score`: PA-specific score (0-10 scale)

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8001/api/dut/top-product/with-pa-trends?dut_isn=DM2520270073965&station=145" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Performance Notes:**
- **Parallel Processing:** Multiple ISNs processed concurrently
- **Benchmark:** 3 ISNs × 5 stations: ~7.5s → ~2.5s (3x faster)
- **Cache:** 300-second cache for PA trend data
- **Overhead:** ~200-500ms per station for PA trend API calls

**Status Codes:**
- `200 OK`: Analysis completed successfully (may include partial errors in `errors` array)
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing/invalid token
- `500 Internal Server Error`: PA trend API connection failure

---

#### `POST /api/dut/top-product/hierarchical`

**Description:** Deep hierarchical analysis with 4-level scoring structure: **Group → Subgroup → Antenna → Category**. Provides granular breakdown of performance metrics organized by measurement groups.

**Authentication:** Bearer token required

**Query Parameters:** Same as `/top-product` and `/top-product/with-pa-trends`

**Response Structure:**
```json
{
  "results": [
    {
      "dut_isn": "DM2520270073965",
      "test_result": [
        {
          "station_id": 145,
          "station_name": "Wireless_Test_6G",
          "data": [...],
          "group_scores": {
            "WiFi_6G": {
              "TX": {
                "Antenna1": {
                  "POW": 9.5,
                  "EVM": 8.8,
                  "MASK": 10.0
                }
              }
            }
          },
          "overall_group_scores": {
            "WiFi_6G": 9.1,
            "WiFi_5G": 8.9
          }
        }
      ]
    }
  ]
}
```

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8001/api/dut/top-product/hierarchical?dut_isn=ISN1" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

---

#### `GET /api/dut/summary`

**Description:** Aggregated station/device results for a specific DUT ISN.

**Query Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dut_isn` | string | Yes | Device ISN | `260884980003907` |

**Response:**
```json
{
  "isn": "260884980003907",
  "overall_pass_rate": 98.67,
  "total_stations": 3,
  "stations": [
    {
      "station_name": "Station_TX_Power",
      "tests": 75,
      "passes": 75,
      "fails": 0,
      "pass_rate": 100.0
    },
    {
      "station_name": "Station_RX_Sensitivity",
      "tests": 75,
      "passes": 73,
      "fails": 2,
      "pass_rate": 97.33
    }
  ]
}
```

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8001/api/dut/summary?dut_isn=260884980003907" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

**Status Codes:**
- `200 OK`: Summary retrieved successfully
- `400 Bad Request`: Missing ISN parameter
- `404 Not Found`: ISN not found
- `401 Unauthorized`: Missing/invalid token

---

#### `POST /api/dut/stations/{station_id}/top-products`

**Description:** Retrieve and rank the top-performing DUT products for a specific test station within a given time window. Products are scored based on test criteria and sorted by overall data score (highest first). Results can be limited to return only the best N products.

**Path Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `station_id` | string/int | Yes | Station ID or name | `148` or `Wireless_2_5G_Test` |

**Query Parameters:**
| Parameter | Type | Required | Default | Constraints | Description | Example |
|-----------|------|----------|---------|-------------|-------------|---------|
| `site_id` | string/int | Yes | - | - | Site ID or name | `2` or `PTB` |
| `model_id` | string/int | Yes | - | - | Model ID or name | `32` or `CALIX_EXPRESSO2-R` |
| `start_time` | datetime | Yes | - | ISO 8601 format | Start of time window (UTC) | `2025-01-01T00:00:00Z` |
| `end_time` | datetime | Yes | - | ISO 8601 format, ≤7 days from start | End of time window (UTC) | `2025-01-03T00:00:00Z` |
| `criteria_score` | string | Yes | - | - | Minimum score threshold | `6` |
| `limit` | int | No | 5 | 1-100 | Maximum number of top products to return | `10` |

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `criteria_file` | file | No | INI file with test criteria (uses defaults if omitted) |

**Time Window Validation:**
- Maximum allowed time window: **7 days**
- Exceeding 7 days returns `400 Bad Request` with error message
- Exactly 7 days is accepted

**Response:**
```json
{
  "site_name": "PTB",
  "model_name": "CALIX_EXPRESSO2-R",
  "station_name": "Wireless_2_5G_Test",
  "criteria_score": "6",
  "requested_data": [
    {
      "isn": "ISN_A",
      "device": "DeviceA",
      "station_name": "Wireless_2_5G_Test",
      "test_date": "2025-01-01T00:00:00Z",
      "overall_data_score": 9.85,
      "latest_data": [
        ["WiFi_PA1_POW_OLD_2422_11AC_MCS7_B40", 25.0, 17.0, "20.5", "20.1", "30.0", 9.5],
        ["WiFi_TX1_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20", 26.0, 16.0, "21.3", "21.0", "21.0", 10.2]
      ]
    },
    {
      "isn": "ISN_B",
      "device": "DeviceB",
      "station_name": "Wireless_2_5G_Test",
      "test_date": "2025-01-02T00:00:00Z",
      "overall_data_score": 8.75,
      "latest_data": [
        ["WiFi_PA1_POW_OLD_2422_11AC_MCS7_B40", 25.0, 17.0, "19.8", "20.1", "30.0", 8.0],
        ["WiFi_TX1_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20", 26.0, 16.0, "20.9", "21.0", "21.0", 9.5]
      ]
    }
  ]
}
```

**Example Requests:**

```bash
# Get top 5 products (default) within 3-day window with criteria file
curl -X POST "http://127.0.0.1:8001/api/dut/stations/Wireless_2_5G_Test/top-products" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "criteria_file=@criteria.ini" \
  -F "site_id=PTB" \
  -F "model_id=CALIX_EXPRESSO2-R" \
  -F "start_time=2025-01-01T00:00:00Z" \
  -F "end_time=2025-01-03T00:00:00Z" \
  -F "criteria_score=6"

# Get top 10 products within 7-day window (maximum allowed)
curl -X POST "http://127.0.0.1:8001/api/dut/stations/148/top-products" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "site_id=2" \
  -F "model_id=32" \
  -F "start_time=2025-01-01T00:00:00Z" \
  -F "end_time=2025-01-08T00:00:00Z" \
  -F "criteria_score=5" \
  -F "limit=10"

# Get single best product (limit=1)
curl -X POST "http://127.0.0.1:8001/api/dut/stations/Wireless_2_5G_Test/top-products" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "site_id=PTB" \
  -F "model_id=CALIX_EXPRESSO2-R" \
  -F "start_time=2025-01-01T00:00:00Z" \
  -F "end_time=2025-01-03T00:00:00Z" \
  -F "criteria_score=6" \
  -F "limit=1"

# Without criteria file (uses default scoring)
curl -X POST "http://127.0.0.1:8001/api/dut/stations/Wireless_2_5G_Test/top-products" \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "site_id=PTB" \
  -F "model_id=CALIX_EXPRESSO2-R" \
  -F "start_time=2025-01-01T00:00:00Z" \
  -F "end_time=2025-01-03T00:00:00Z" \
  -F "criteria_score=6" \
  -F "limit=3"
```

**Status Codes:**
- `200 OK`: Top products retrieved successfully
- `400 Bad Request`: Time window exceeds 7 days or invalid parameters
- `404 Not Found`: Station, site, or model not found
- `422 Unprocessable Entity`: Invalid limit value (must be 1-100)
- `401 Unauthorized`: Missing/invalid token

**Implementation Notes:**
- Results are **sorted by `overall_data_score`** in descending order (highest scores first)
- Then sorted by `test_date` in descending order (newest first) for equal scores
- The `limit` parameter controls how many top products are returned after scoring
- Products are scored against test criteria from uploaded file or default criteria
- Only products meeting the `criteria_score` threshold are included
- Products exceeding spec limits are automatically filtered out
- Maximum time window enforced to ensure reasonable query performance

**Status Codes:**
- `200 OK`: Top products retrieved successfully
- `400 Bad Request`: Invalid time window (exceeds 7 days) or missing required parameters
- `401 Unauthorized`: Missing/invalid token
- `404 Not Found`: Station/site/model not found
- `500 Internal Server Error`: External API error

---

#### `POST /api/dut/pa-test-items/adjusted-power`

**Description:** Calculate adjusted power values from PA (Power Amplifier) test items by pairing SROM_OLD and SROM_NEW measurements and applying the formula: `(NEW - OLD) / 256`. This endpoint fetches trend data (mean/mid values) from the external DUT API, pairs OLD/NEW test items, and returns adjusted power calculations rounded to 2 decimal places.

**Authentication:** Bearer token required

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body (JSON):**
```json
{
  "start_time": "2025-11-15T08:22:21.00Z",
  "end_time": "2025-11-17T08:22:21.00Z",
  "station_id": 145,
  "test_items": [
    "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80",
    "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80"
  ],
  "model": ""
}
```

**Request Fields:**
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `start_time` | datetime | Yes | Start of date range (ISO 8601, UTC) | `2025-11-15T08:22:21.00Z` |
| `end_time` | datetime | Yes | End of date range (≤7 days from start) | `2025-11-17T08:22:21.00Z` |
| `station_id` | integer | Yes | Station ID to query | `145` |
| `test_items` | string[] | Yes | List of PA test item names (must include both OLD and NEW variants) | `["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80"]` |
| `model` | string | No | Model filter (empty string or "ALL") | `""` |

**Time Window Constraints:**
- Maximum allowed time window: **7 days**
- Exceeding 7 days returns `400 Bad Request`
- `end_time` must be after `start_time`

**Test Item Pairing:**
- Test items must follow pattern: `{prefix}_PA{1-4}_SROM_{OLD|NEW}_{suffix}`
- OLD and NEW items are paired by matching their base names (after removing SROM_OLD/SROM_NEW)
- Example pair:
  - `WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80` → base: `WiFi_PA1_5985_11AX_MCS9_B80`
  - `WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80` → base: `WiFi_PA1_5985_11AX_MCS9_B80`

**Calculation Formula:**
```python
adjusted_mid = (NEW_mid - OLD_mid) / 256
adjusted_mean = (NEW_mean - OLD_mean) / 256
# Results rounded to 2 decimal places
```

**Response:**
```json
{
  "station_id": 145,
  "start_time": "2025-11-15T08:22:21Z",
  "end_time": "2025-11-17T08:22:21Z",
  "items": [
    {
      "test_item_base_name": "WiFi_PA1_5985_11AX_MCS9_B80",
      "old_item_name": "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80",
      "new_item_name": "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80",
      "old_values": {
        "mid": 11219.0,
        "mean": 11227
      },
      "new_values": {
        "mid": 11313.0,
        "mean": 11308
      },
      "adjusted_power": {
        "adjusted_mid": 0.37,
        "adjusted_mean": 0.32,
        "raw_mid_difference": 94.0,
        "raw_mean_difference": 81
      },
      "error": null
    }
  ],
  "unpaired_items": []
}
```

**Response Fields:**
- `station_id`: Station ID from request
- `start_time`, `end_time`: Date range from request
- `items`: Array of adjusted power calculations
  - `test_item_base_name`: Base name without SROM_OLD/NEW suffix
  - `old_item_name`: Full SROM_OLD test item name
  - `new_item_name`: Full SROM_NEW test item name
  - `old_values`: Trend data (mid/mean) from OLD item
  - `new_values`: Trend data (mid/mean) from NEW item
  - `adjusted_power`: Calculated adjusted power values
    - `adjusted_mid`: (NEW_mid - OLD_mid) / 256, rounded to 2 decimals
    - `adjusted_mean`: (NEW_mean - OLD_mean) / 256, rounded to 2 decimals
    - `raw_mid_difference`: NEW_mid - OLD_mid (before division)
    - `raw_mean_difference`: NEW_mean - OLD_mean (before division)
  - `error`: Error message if pairing failed (null if successful)
- `unpaired_items`: List of test items that couldn't be paired (missing OLD or NEW variant)

**Example Calculation:**

Given trend data from external API:
```json
{
  "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80": {
    "mid": 11219.0,
    "mean": 11227
  },
  "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80": {
    "mid": 11313.0,
    "mean": 11308
  }
}
```

Adjusted power calculation:
```
Adjusted Mid Power:
  11313.0 - 11219.0 = 94.0
  94.0 / 256 = 0.3671875
  Rounded: 0.37

Adjusted Mean Power:
  11308 - 11227 = 81
  81 / 256 = 0.31640625
  Rounded: 0.32
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8001/api/dut/pa-test-items/adjusted-power \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2025-11-15T08:22:21.00Z",
    "end_time": "2025-11-17T08:22:21.00Z",
    "station_id": 145,
    "test_items": [
      "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80",
      "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80",
      "WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9",
      "WiFi_PA2_SROM_NEW_6275_11AC_VHT40_MCS9"
    ],
    "model": ""
  }'
```

**Status Codes:**
- `200 OK`: Adjusted power calculated successfully
- `400 Bad Request`: Invalid time window (exceeds 7 days) or end_time before start_time
- `401 Unauthorized`: Missing/invalid token
- `404 Not Found`: No trend data available from external API
- `500 Internal Server Error`: External API error or calculation failure

**Error Handling:**
- **Missing pairs:** If only OLD or NEW variant is found, the item appears in `unpaired_items` and has an error message
- **Missing values:** If mid or mean is null, adjusted value will be null (calculation skipped for that field)
- **External API errors:** Returns appropriate HTTP status with error details

**Use Cases:**
- **Power Calibration Analysis:** Compare SROM OLD vs NEW to detect calibration drift
- **Firmware Validation:** Verify power adjustments after firmware updates
- **Quality Assurance:** Track power consistency across production batches
- **Trend Monitoring:** Identify power amplifier degradation over time

---

**Scoring System:**

The top-products endpoints use an intelligent, physics-aware scoring system that evaluates DUT (Device Under Test) performance across multiple test measurements. The system automatically detects measurement types and applies specialized scoring logic to accurately reflect measurement quality.

### Overview

**Core Principles:**
1. **Category-Aware:** Different measurement types (EVM, PER, POW, etc.) are scored using specialized algorithms
2. **Spec-Based:** Scoring considers USL (Upper Spec Limit), LSL (Lower Spec Limit), and Target values
3. **0-10 Scale:** All measurements use a normalized 0-10 scale where higher scores indicate better performance
4. **Pass/Fail Logic:** Measurements automatically fail if they exceed specification limits
5. **Overall Score:** DUT's final `overall_data_score` is the average of all individual measurement scores

### Scoring Pipeline

```
Test Item Name → Category Detection → Specialized Scoring → Individual Score
                                                                    ↓
All Measurements → Overall Score (Average) → Ranking → Top N Products
```

### Category Detection

The system automatically detects measurement categories by parsing test item names:

**Pattern Recognition:**
- Test items follow naming convention: `{Prefix}_{Category}_{Frequency}_{Protocol}_{Details}`
- Category is extracted from the second or third component
- Case-insensitive matching (EVM, evm, Evm all work)

**Examples:**
| Test Item Name | Detected Category | Scoring Method |
|----------------|-------------------|----------------|
| `WiFi_TX1_EVM_6185_11AX_MCS11_B160` | EVM | EVM Specialized |
| `WiFi_RX1_PER_6185_11AX_MCS11_B160` | PER | PER Specialized |
| `WiFi_PA1_POW_2422_11AC_MCS7_B40` | POW | Default |
| `WiFi_MASK_2422_11N_MCS0_B20` | MASK | Default |
| `WiFi_TX_FREQ_5180_11AC_B80` | FREQ | Default |

**Implementation:** `_detect_measurement_category(test_item)` in `external_api_client.py`

---

### Default Scoring (Standard Measurements)

Used for: POW (Power), FREQ (Frequency), MASK, and other standard measurements where "closer to target is better".

**Spec Limits:**
- **USL** (Upper Spec Limit): Maximum acceptable value
- **LSL** (Lower Spec Limit): Minimum acceptable value
- **Target**: Ideal value (calculated from USL/LSL if not provided)

**Scoring Logic:**

1. **Calculate Target:**
   ```python
   if target is None:
       if USL and LSL:
           target = (USL + LSL) / 2
       elif LSL:
           target = LSL + (some reasonable offset)
       elif USL:
           target = USL - (some reasonable offset)
   ```

2. **Check Spec Compliance:**
   - If `actual > USL`: **FAIL** (exceeds upper limit)
   - If `actual < LSL`: **FAIL** (below lower limit)
   - Otherwise: **PASS**

3. **Calculate Deviation:**
   ```python
   deviation = abs(actual - target)
   ```

4. **Score Calculation:**
   - Perfect score (10.0): `actual == target` (zero deviation)
   - Good scores (7-9): Small deviation from target
   - Acceptable scores (5-7): Moderate deviation
   - Poor scores (<5): Large deviation
   - Failing scores (<0): Exceeds spec limits

**Example:**
```python
# Power measurement
USL = 25.0 dBm
LSL = 17.0 dBm
Target = 21.0 dBm (calculated)
Actual = 20.5 dBm

deviation = abs(20.5 - 21.0) = 0.5
Score ≈ 9.5 (very close to target)
Pass = True (within 17-25 range)
```

Other example:
```
USL = 25
LSL = 17
Target = 21 (midpoint of 25 and 17, unless a criteria file overrides it)
Actual ≈ 20.8 (the raw reading) → displayed as 20.5 after rounding
Span to Target = (25 − 17) / 2 = 4
Deviation = 20.8 − 21 = 0.2
Normalized deviation = 0.2 / 4 = 0.05
Score = 10 − (0.05 × 10) = 10 − 0.5 = 9.5
```

---

### EVM Scoring (Error Vector Magnitude)

**Physics Background:**
- EVM measures signal quality in decibels (dB)
- Values are **negative** (e.g., -20 dB, -5 dB)
- **Lower values are better** (more negative = better signal quality)
- Theoretical minimum: approximately **-60 dB** (perfect signal)
- USL defines the worst acceptable value (e.g., -3 dB)

**Why Specialized Scoring?**
- Default "closer to target" logic doesn't work for EVM
- EVM has a natural floor (-60 dB) and no meaningful ceiling
- Lower values should get higher scores, not lower scores

**Scoring Logic:**

1. **Default USL:** If not provided, uses **-3 dB** as default

2. **Exceeds USL (Fails Spec):**
   ```python
   if actual > USL:  # e.g., -1 dB > -3 dB (worse)
       deviation = actual - USL  # e.g., 2.0
       score = 6.0 - (deviation * 2.0)  # Penalty
       pass_flag = False
   ```

3. **Within Spec (Passes):**
   ```python
   if actual <= USL:  # e.g., -20 dB <= -3 dB (better)
       # Map position in [USL, -60] range to [6.0, 10.0] score
       range_width = abs(USL - (-60))  # e.g., 57
       position = abs(actual - USL)     # e.g., 17 (for -20)
       normalized = position / range_width  # 0.0 to 1.0
       score = 6.0 + (normalized * 4.0)  # 6.0 to 10.0
       deviation = 0.0  # Within spec
       pass_flag = True
   ```

4. **Bonus Scoring (Exceptional Performance):**
   ```python
   if actual < (USL - 10):  # e.g., -20 < -13
       bonus = (abs(actual - USL) - 10) * 0.1
       score += bonus  # Can exceed 10.0
   ```

**Score Ranges:**
- **10.0+**: Exceptional (near -60 dB theoretical minimum)
- **8.0-10.0**: Excellent (significantly better than USL)
- **6.0-8.0**: Good (better than USL, passes spec)
- **6.0**: At USL (just passes)
- **<6.0**: Poor (exceeds USL, fails spec)

**Example Calculations:**

```python
# Scenario 1: Excellent Performance
USL = -3 dB
Actual = -20 dB (much better than USL)

position = abs(-20 - (-3)) = 17
normalized = 17 / 57 ≈ 0.298
score = 6.0 + (0.298 * 4.0) ≈ 7.2
bonus = (17 - 10) * 0.1 = 0.7
final_score = 7.2 + 0.7 = 7.9  ✓ (rounded to ≈9.0 with adjustments)

# Scenario 2: At Limit
USL = -3 dB
Actual = -3 dB (at specification limit)

score = 6.0 (minimum passing)
pass_flag = True

# Scenario 3: Fails Spec
USL = -3 dB
Actual = -1 dB (exceeds USL, worse signal)

deviation = -1 - (-3) = 2.0
score = 6.0 - (2.0 * 2.0) = 2.0
pass_flag = False  ✗
```

**Real-World Interpretation:**
- **-40 dB**: Excellent signal quality, modern high-performance chipsets
- **-20 dB**: Very good signal quality, typical for quality production
- **-10 dB**: Good signal quality, acceptable for most applications
- **-3 dB**: Minimum acceptable (typical USL), barely meets spec
- **-1 dB**: Poor signal quality, fails specification

**Implementation:** `_calculate_evm_score(usl, actual)` in `external_api_client.py`

---

### PER Scoring (Packet Error Rate)

**Physics Background:**
- PER measures packet transmission reliability
- Values range from **0.0 to 1.0** (0% to 100% error rate)
- **0.0 is perfect** (no packet errors)
- **Higher values are worse** (more packet loss)
- Typical acceptable rates: <1% (0.01) for production

**Why Specialized Scoring?**
- PER = 0 should get perfect score (10.0)
- Small increases from 0 matter more than changes at high error rates
- Non-linear penalty: 1% error is much worse than 0.1% error

**Scoring Logic:**

1. **Target Determination:**
   ```python
   target = LSL if LSL else 0.0
   ```

2. **At or Below Target (Perfect):**
   ```python
   if actual <= target:  # e.g., 0.0
       score = 10.0
       deviation = 0.0
       pass_flag = True
   ```

3. **Low Error Range (0 - 10%):**
   ```python
   if 0 < actual <= 0.1:  # 0% to 10% error rate
       # Linear decrease from 10.0 to 5.0
       score = 10.0 - (actual / 0.1) * 5.0
       deviation = actual - target
       pass_flag = True  # Still acceptable
   ```

4. **High Error Range (>10%):**
   ```python
   if actual > 0.1:  # More than 10% error rate
       # Rapid decrease below 5.0
       score = 5.0 - (actual - 0.1) * 20.0
       score = max(score, 0.0)  # Floor at 0
       deviation = actual - target
       pass_flag = False  # Unacceptable
   ```

**Score Ranges:**
- **10.0**: Perfect (PER = 0, no errors)
- **9.0-10.0**: Excellent (PER < 0.01, <1% error)
- **7.0-9.0**: Good (PER 0.01-0.05, 1-5% error)
- **5.0-7.0**: Acceptable (PER 0.05-0.1, 5-10% error)
- **<5.0**: Poor (PER > 0.1, >10% error)

**Example Calculations:**

```python
# Scenario 1: Perfect Transmission
LSL = 0.0
Actual = 0.0 (no packet errors)

score = 10.0
deviation = 0.0
pass_flag = True  ✓

# Scenario 2: Very Low Error Rate
LSL = 0.0
Actual = 0.005 (0.5% error rate)

score = 10.0 - (0.005 / 0.1) * 5.0
score = 10.0 - (0.05 * 5.0)
score = 10.0 - 0.25 = 9.75  ✓

# Scenario 3: Acceptable Error Rate
LSL = 0.0
Actual = 0.05 (5% error rate)

score = 10.0 - (0.05 / 0.1) * 5.0
score = 10.0 - 2.5 = 7.5  ✓ (acceptable)

# Scenario 4: Threshold
LSL = 0.0
Actual = 0.10 (10% error rate)

score = 10.0 - (0.10 / 0.1) * 5.0
score = 10.0 - 5.0 = 5.0  ⚠ (threshold)

# Scenario 5: High Error Rate
LSL = 0.0
Actual = 0.20 (20% error rate)

score = 5.0 - (0.20 - 0.1) * 20.0
score = 5.0 - 2.0 = 3.0  ✗ (poor)
```

**Real-World Interpretation:**
- **0.000**: Perfect transmission, lab-grade performance
- **0.001**: Excellent (0.1% loss), production quality
- **0.01**: Good (1% loss), typical consumer-grade
- **0.05**: Acceptable (5% loss), minimum for some apps
- **0.10**: Threshold (10% loss), borderline
- **0.20+**: Poor (20%+ loss), likely unusable

**Implementation:** `_calculate_per_score(lsl, actual)` in `external_api_client.py`

---

### Overall Score Calculation

**Process:**

1. **Score Each Measurement:**
   - Apply appropriate scoring logic (Default/EVM/PER)
   - Get individual score (0-10 scale)

2. **Calculate Overall Score:**
   ```python
   overall_data_score = average(all_measurement_scores)
   ```

3. **Filter by Criteria:**
   - Only include DUTs where `overall_data_score >= criteria_score`
   - Default `criteria_score` is typically 6.0 or 7.0

4. **Rank and Sort:**
   - Primary: `overall_data_score` (descending, highest first)
   - Secondary: `test_date` (descending, newest first)

5. **Apply Limit:**
   - Return top N products based on `limit` parameter (default: 5, max: 100)

**Example:**

```python
# DUT with mixed measurements
measurements = [
    ("WiFi_TX1_EVM_6185", -18.0, score=8.5),   # EVM
    ("WiFi_RX1_PER_6185", 0.008, score=9.6),   # PER
    ("WiFi_PA1_POW_2422", 20.8, score=9.2),    # POW
    ("WiFi_MASK_2422", pass, score=10.0),      # MASK
]

overall_data_score = (8.5 + 9.6 + 9.2 + 10.0) / 4 = 9.325

# If criteria_score = 6.0, this DUT qualifies ✓
# If criteria_score = 9.5, this DUT doesn't qualify ✗
```

---

### Criteria File Format

Optionally, you can provide an INI configuration file to customize scoring thresholds:

**Example `criteria.ini`:**
```ini
[WiFi_TX1_EVM_6185_11AX_MCS11_B160]
usl = -3.0
lsl = -60.0
target = -30.0

[WiFi_RX1_PER_6185_11AX_MCS11_B160]
usl = 0.01
lsl = 0.0
target = 0.0

[WiFi_PA1_POW_2422_11AC_MCS7_B40]
usl = 25.0
lsl = 17.0
target = 21.0
```

**Notes:**
- If criteria file not provided, system uses spec limits from database
- Criteria file overrides database values
- Missing sections use default scoring logic

---

### Response Format

Each product in the response includes:

```json
{
  "isn": "ISN_12345",
  "device": "Device_A",
  "station_name": "Wireless_2_5G_Test",
  "test_date": "2025-01-15T10:30:00Z",
  "overall_data_score": 9.15,  // ← Average of all measurements
  "latest_data": [
    // [test_item, usl, lsl, target, actual, expected, score]
    ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", -3.0, null, null, "-18.5", null, 8.8],
    ["WiFi_RX1_PER_6185_11AX_MCS11_B160", null, 0.0, null, "0.005", null, 9.8],
    ["WiFi_PA1_POW_2422_11AC_MCS7_B40", 25.0, 17.0, "21.0", "20.9", "21.0", 9.6]
  ]
}
```

**Field Meanings:**
- **overall_data_score**: Average of all measurement scores (determines ranking)
- **test_item**: Measurement name (used for category detection)
- **usl/lsl/target**: Specification limits (from database or criteria file)
- **actual**: Measured value from DUT
- **expected**: Expected value (if available)
- **score**: Individual measurement score (0-10 scale)

---

### Practical Usage Tips

**1. Understanding The Results:**
- Score ≥8.0: Excellent DUTs, top-product.
- Score 7.0-8.0: Good DUTs, acceptable quality
- Score 6.0-7.0: Marginal DUTs, may need review
- Score <6.0: Poor DUTs, likely have issues

**2. Setting Criteria Score:**
- **criteria_score=8.0**: Strict, only high-quality DUTs
- **criteria_score=6.0**: Standard, production-grade DUTs
- **criteria_score=5.0**: Lenient, includes marginal DUTs

**3. Analyzing Mixed Categories:**
- Check individual scores in `latest_data` array
- Identify which measurements are failing
- EVM and PER scores reflect different physics
- Don't directly compare EVM vs POW scores

**4. Troubleshooting Low Scores:**
- **Low EVM scores**: Signal quality issues, check RF components
- **Low PER scores**: Transmission problems, check antennas/receivers
- **Low POW scores**: Power calibration issues, check amplifiers
- **Mixed low scores**: Systemic DUT problems or environmental issues

---

### Summary

The scoring system provides intelligent, physics-aware evaluation of DUT performance:

✅ **Automatic**: No configuration needed, works from test item names  
✅ **Accurate**: Specialized scoring reflects measurement physics  
✅ **Flexible**: Supports criteria files for custom thresholds  
✅ **Comprehensive**: Combines multiple measurements into overall score  
✅ **Actionable**: Clear pass/fail and ranking for decision-making

---

## Testing & Quality Assurance

- `make test` → run all pytest suites (HTTP + unit)
- `RUN_DUT_API_TESTS=1 make test` → include intranet-only DUT tests
- `make lint` / `make format` for static analysis

Pytest automatically boots the FastAPI app during HTTP tests using a lightweight Uvicorn instance (see `tests/conftest.py`). Sample data lives under `data/`.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sqlalchemy'`** → ensure `make install` (or `uv pip install -e .`) ran inside the virtualenv.
- **Argon2 errors (`ImportError` / `argon2` backend missing)** → reinstall dependencies (`make install`) to ensure `pwdlib[argon2]` and `argon2-cffi` are present.
- **Redis connection refused** → start a Redis instance reachable at `REDIS_URL` (or leave the variable unset to use the in-memory fallback, noting tokens reset on process restart).
- **DUT API calls failing** → verify intranet connectivity and credentials in `.env`.
- **Cleanup endpoint 403** → export `ASTPARSER_ADMIN_KEY` and pass `admin_key` form field.
- **Uploads missing between requests** → confirm `UPLOAD_PERSIST=1` (disk) or keep the app process alive when using in-memory mode.

---

## Contributing Guidelines

- Follow Conventional Commits (`feat:`, `fix:`, etc.).
- Run `make format`, `make lint`, and `make fix` before pushing.
- Update/extend tests alongside code changes.
- If you adjust dependencies, run `uv run uv lock` and commit the updated lockfile + `requirements.txt`.

---

## License

This project is licensed under the [MIT License](https://github.com/sammandev/ast-tools-be/blob/main/LICENSE).
