# AST Parser - Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites

1. **Docker & Docker Compose** installed on staging server
2. **Git** for cloning the repository
3. **Access to PostgreSQL database** at `172.18.220.56:5432`
4. **Network access** to DUT API at `172.18.220.56:9001`

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd backend_fastapi
   ```

2. **Configure environment:**
   ```bash
   cp .env.staging .env
   # Edit .env and set JWT_SECRET to a secure random string
   nano .env
   ```

3. **Generate secure JWT secret:**
   ```bash
   # Linux/Mac
   openssl rand -hex 32
   
   # PowerShell
   [System.Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
   ```
   
   Update `JWT_SECRET` in `.env` with the generated value.

4. **Deploy the application:**
   ```bash
   # Linux/Mac
   chmod +x deploy.sh
   ./deploy.sh staging
   
   # Windows PowerShell
   .\deploy.ps1 staging
   ```

### Manual Deployment

If deployment scripts don't work:

```bash
# Build images
docker-compose build --no-cache

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

---

## 📋 Configuration Files

### Backend Configuration

**`.env` file** (copy from `.env.staging`):
```env
# Application
APP_NAME=AST_Parser_Staging
DEBUG=False
ENVIRONMENT=staging

# Database (Hosted PostgreSQL)
DATABASE_URL=postgresql+psycopg://test:test1234@172.18.220.56:5432/test123

# JWT Security
JWT_SECRET=<GENERATE_SECURE_RANDOM_STRING_HERE>

# External DUT API
DUT_API_BASE_URL=http://172.18.220.56:9001

# CORS
CORS_ORIGINS=http://172.18.220.56,http://172.18.220.56:3000
```

### Frontend Configuration

**`.env.production`** (already configured):
```env
VITE_API_BASE_URL=http://172.18.220.56:9090/api
VITE_APP_TITLE=AST Parser - Staging
VITE_ENVIRONMENT=staging
```

---

## 🏗️ Architecture Overview

### Docker Services

1. **Backend (FastAPI)**
   - Port: 7070 (internal)
   - Workers: 4
   - Health check: `/health` endpoint
   - Connected to external PostgreSQL database

2. **Redis**
   - Port: 6379
   - Used for caching and token storage
   - Persistent volume

3. **Nginx**
   - Port: 80 (HTTP)
   - Serves frontend static files
   - Reverse proxy for backend API
   - Rate limiting and security headers

### Network Architecture

```
Internet/Intranet
       ↓
  Nginx :80 (172.18.220.56)
       ├─→ Frontend (Vue.js SPA)
       └─→ /api/* → Backend :7070
                       ↓
                ┌──────┴──────┐
                ↓             ↓
         PostgreSQL      Redis
      (172.18.220.56)   (local)
                ↓
           DUT API
      (172.18.220.56:9001)
```

---

## 🔒 Security Configuration

### CORS Settings

Backend allows requests from:
- `http://172.18.220.56` (production frontend)
- `http://172.18.220.56:3000` (development frontend)
- `http://localhost:3000` (local development)

### Rate Limiting (Nginx)

- API endpoints: 30 requests/second per IP
- Auth endpoints: 5 requests/second per IP
- Burst: 50 requests
- Connection limit: 10 concurrent per IP

### Security Headers

- `X-Frame-Options: SAMEORIGIN`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: no-referrer-when-downgrade`

---

## 📊 Database Setup

The application uses an **existing PostgreSQL database**. No database container needed.

**Database credentials:**
```
Host: 172.18.220.56
Port: 5432
Database: test123
Username: test
Password: test1234
```

### Initial Database Setup

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Seed RBAC (roles, permissions, default users)
docker-compose exec backend python scripts/bootstrap_rbac.py \
  --admin-password "SecureAdminPass!123" \
  --analyst-password "SecureAnalystPass!123" \
  --viewer-password "SecureViewerPass!123"
```

### Database Migrations

```bash
# Check current migration version
docker-compose exec backend alembic current

# Upgrade to latest
docker-compose exec backend alembic upgrade head

# Rollback one version
docker-compose exec backend alembic downgrade -1

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "description"
```

---

## 🔧 Maintenance Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f nginx
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Restart Services

```bash
# All services
docker-compose restart

# Specific service
docker-compose restart backend
docker-compose restart nginx

# Rebuild and restart
docker-compose up -d --build backend
```

### Shell Access

```bash
# Backend container
docker-compose exec backend bash

# Run Python commands
docker-compose exec backend python -c "from app.db.init_db import init_db; init_db()"

# Run scripts
docker-compose exec backend python scripts/create_user.py --username admin --password "Secure123!" --admin
```

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Run migrations
docker-compose exec backend alembic upgrade head
```

---

## 📈 Monitoring & Health Checks

### Health Check Endpoints

```bash
# Backend health
curl http://172.18.220.56/health

# API documentation
curl http://172.18.220.56/api/docs

# Frontend
curl http://172.18.220.56
```

### Service Status

```bash
# Docker services
docker-compose ps

# Resource usage
docker stats

# Container health
docker inspect --format='{{.State.Health.Status}}' ast-tools-backend
```

### Log Monitoring

```bash
# Backend errors
docker-compose logs backend | grep ERROR

# Nginx access logs
docker-compose exec nginx tail -f /var/log/nginx/access.log

# Nginx error logs
docker-compose exec nginx tail -f /var/log/nginx/error.log
```

---

## 🐛 Troubleshooting

### Backend Issues

**Problem: Backend not starting**
```bash
# Check logs
docker-compose logs backend

# Check database connection
docker-compose exec backend python -c "from app.db import engine; print(engine.pool.status())"

# Verify environment
docker-compose exec backend env | grep DATABASE_URL
```

**Problem: Database connection failed**
```bash
# Test connection from backend container
docker-compose exec backend psql postgresql://test:test1234@172.18.220.56:5432/test123

# Check network
docker-compose exec backend ping 172.18.220.56
```

### Frontend Issues

**Problem: API calls failing**
```bash
# Check Nginx proxy
docker-compose logs nginx | grep proxy

# Test backend from Nginx
docker-compose exec nginx curl http://backend:7070/health

# Verify CORS headers
curl -H "Origin: http://172.18.220.56" -I http://172.18.220.56/api/health
```

**Problem: 404 on frontend routes**
- Ensure Nginx config has `try_files $uri $uri/ /index.html;`
- Check frontend build includes all routes

### Redis Issues

**Problem: Redis connection failed**
```bash
# Check Redis
docker-compose exec redis redis-cli ping

# Check connections
docker-compose exec redis redis-cli CLIENT LIST

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

### Common Fixes

**Clear everything and restart:**
```bash
docker-compose down -v
docker-compose up -d --build
```

**Reset database (CAUTION - deletes data):**
```bash
docker-compose exec backend alembic downgrade base
docker-compose exec backend alembic upgrade head
docker-compose exec backend python scripts/bootstrap_rbac.py
```

---

## 🔐 SSL/HTTPS Setup (Optional)

### Enable HTTPS

1. **Generate SSL certificates:**
   ```bash
   mkdir -p nginx/ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem \
     -out nginx/ssl/cert.pem
   ```

2. **Update Nginx config:**
   Uncomment SSL sections in `nginx/conf.d/default.conf`

3. **Restart Nginx:**
   ```bash
   docker-compose restart nginx
   ```

### Let's Encrypt (Production)

For production with domain name:

1. Install certbot
2. Generate certificates
3. Update Nginx to use certificates
4. Set up auto-renewal

---

## 📝 Backup & Recovery

### Database Backup

```bash
# Backup database
docker-compose exec backend pg_dump \
  postgresql://test:test1234@172.18.220.56:5432/test123 \
  > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
docker-compose exec -T backend psql \
  postgresql://test:test1234@172.18.220.56:5432/test123 \
  < backup_20251209_120000.sql
```

### Volume Backup

```bash
# Backup uploads
docker run --rm -v backend_fastapi_upload_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/uploads_backup.tar.gz -C /data .

# Restore uploads
docker run --rm -v backend_fastapi_upload_data:/data \
  -v $(pwd):/backup alpine tar xzf /backup/uploads_backup.tar.gz -C /data
```

---

## 🎯 Production Checklist

Before going to production:

- [ ] Change `DEBUG=False` in `.env`
- [ ] Generate strong `JWT_SECRET` (32+ characters)
- [ ] Update `CORS_ORIGINS` to production domains only
- [ ] Enable HTTPS/SSL
- [ ] Set up database backups
- [ ] Configure log rotation
- [ ] Set up monitoring (Sentry, etc.)
- [ ] Review and harden Nginx security
- [ ] Test all API endpoints
- [ ] Run full test suite
- [ ] Load testing
- [ ] Document runbooks
- [ ] Set up alerting

---

## 📞 Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review this documentation
3. Check GitHub issues
4. Contact development team

---

**Last Updated:** December 9, 2025
