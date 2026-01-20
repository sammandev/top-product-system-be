# Shell Scripts for Remote Server Deployment

This directory contains shell scripts for managing Docker-based deployments of the AST Tools backend from a remote Ubuntu server.

## 🏗️ Architecture

The staging setup uses:
- **Hot reload enabled**: Source code is mounted as a volume, changes are picked up automatically
- **Host network mode**: Container uses the host's network to access PostgreSQL and Redis
- **No container Redis**: Uses existing Redis on the host (configured via `REDIS_URL` in `.env.staging`)

## 📋 Available Scripts

### 1. `deploy-staging.sh` - Initial Deployment / Rebuild
Full deployment with Docker image build. Use for initial setup or when dependencies change.

```bash
# Build and deploy (initial deployment or after dependency changes)
./deploy-staging.sh

# Start without rebuilding (just restart services)
./deploy-staging.sh --no-build
```

**When to use:**
- ✅ Initial deployment
- ✅ After changing `pyproject.toml` (dependencies)
- ✅ After changing `Dockerfile.staging`

**When NOT to use:**
- ❌ Just updating Python code (use `update-code.sh` instead - faster!)

---

### 2. `update-code.sh` - Update Code (No Rebuild!)
Restart services to pick up code changes and run migrations. **Does NOT rebuild or recreate containers.**

```bash
# First, manually pull latest code
git pull

# Then restart the service and run migrations
./update-code.sh

# Skip migrations if not needed
./update-code.sh --skip-migrations
```

**What it does:**
- ♻️ Restarts the service (lightweight, no new images/containers)
- 🔄 Runs `alembic upgrade head` inside the container
- ⏳ Health check verification

**Why no rebuild?**
- Source code is mounted as a volume (`./src:/app/src`)
- Uvicorn runs with `--reload` flag
- `docker compose restart` is lightweight - just sends restart signal
- Migrations run separately via `docker compose exec`

**Note:** Remember to run `git pull` manually before running this script!

---

### 3. `refresh-env.sh` - Apply Environment Variable Changes
Restart services to pick up new environment variables. **Does NOT rebuild.**

```bash
# 1. Edit .env.staging file
nano .env.staging

# 2. Apply changes
./refresh-env.sh
```

**What it does:**
- 🔍 Verifies `.env.staging` exists
- ♻️ Restarts service to pick up new env vars
- 📊 Shows current environment settings

---

### 4. `run-migrations-docker.sh` - Run Database Migrations
Apply Alembic migrations inside the running Docker container.

```bash
./run-migrations-docker.sh
```

**What it does:**
- 📋 Shows current migration status
- 🔍 Checks for unapplied migrations
- 🚀 Applies pending migrations
- ✅ Verifies migrations were applied

**Note:** Migrations also run automatically on container startup (via the `command` in docker-compose).

---

### 5. `backup_db.sh` - Backup Database
Create a compressed backup of the PostgreSQL database.

```bash
./backup_db.sh
```

**Output:** `backups/backup_<dbname>_<timestamp>.sql.gz`

---

### 6. `restore_db.sh` - Restore Database
Restore database from a backup file.

```bash
# List available backups
./restore_db.sh

# Restore specific backup
./restore_db.sh ./backups/backup_top_products_db_20260119_143000.sql.gz
```

**Warning:** This REPLACES the current database!

---

### 7. `make-migration.sh` - Create New Migration (Development Only)
Generate a new Alembic migration based on model changes.

```bash
./make-migration.sh "add user worker_id column"
```

---

## 🔄 Common Workflows

### Deploy New Code (Daily Usage)
```bash
# 1. Pull latest code manually
git pull

# 2. Recreate container (runs migrations on startup)
./update-code.sh
```

### Update Environment Variables
```bash
# 1. Edit .env.staging file
nano .env.staging

# 2. Restart to apply
./refresh-env.sh
```

### Initial Deployment / Dependency Changes
```bash
# Full rebuild
./deploy-staging.sh
```

### Run Migrations Manually
```bash
./run-migrations-docker.sh
```

### Backup Before Major Changes
```bash
# 1. Create backup
./backup_db.sh

# 2. Make changes
./update-code.sh

# 3. If something goes wrong
./restore_db.sh ./backups/backup_top_products_db_20260119_143000.sql.gz
```

---

## 🎯 Key Differences from Build-Based Deployments

| Action | Old Way (Rebuild) | New Way (Hot Reload) |
|--------|-------------------|----------------------|
| Code change | `docker compose up --build` | `./update-code.sh` (just restart) |
| Env change | Recreate container | `./refresh-env.sh` (just restart) |
| Dependency change | Rebuild required | `./deploy-staging.sh` (rebuild) |
| Time for code deploy | ~1-2 minutes | ~5-10 seconds |

---

## 🔧 Configuration

### Docker Compose Settings
- **Network mode**: `host` (accesses PostgreSQL/Redis on host network)
- **Hot reload**: Enabled via `uvicorn --reload`
- **Volume mounts**: `./src:/app/src` for code hot reload

### Environment Variables (in `.env.staging`)
```bash
# Database (external PostgreSQL)
DB_HOST=172.18.220.56
DB_PORT=5432
DB_NAME=top_products_db
DB_USER=ptb
DB_PASSWORD=ptb#1234

# Redis (external, on host)
REDIS_URL=redis://172.18.220.56:7071/0
```

---

## 🐛 Troubleshooting

### "Connection timeout expired" (Database)
The container can't reach PostgreSQL. Check:
1. Is PostgreSQL running? `sudo systemctl status postgresql`
2. Is the IP correct in `.env.staging`?
3. Is the firewall allowing connections?

### "Redis connection refused"
1. Is Redis running on the host? `redis-cli -p 7071 ping`
2. Is the `REDIS_URL` correct in `.env.staging`?

### Container not starting
```bash
# Check logs
docker compose -f docker-compose.staging.yml logs backend

# Check container status
docker compose -f docker-compose.staging.yml ps
```

### Health check failing
```bash
# Test health endpoint manually
curl http://localhost:7070/health

# Check what's listening on port 7070
sudo netstat -tlnp | grep 7070
```

---

## 📁 Directory Structure

```
backend_fastapi/
├── shell-scripts/
│   ├── deploy-staging.sh      # Initial deploy (with build)
│   ├── update-code.sh         # Code update (no build!)
│   ├── refresh-env.sh         # Env update (no build!)
│   ├── run-migrations-docker.sh
│   ├── backup_db.sh
│   ├── restore_db.sh
│   ├── make-migration.sh
│   └── README.md
├── docker-compose.staging.yml
├── Dockerfile.staging
├── .env.staging
├── src/                       # Mounted as volume for hot reload
│   └── app/
├── alembic/                   # Mounted for migrations
└── backups/                   # Database backups
```
