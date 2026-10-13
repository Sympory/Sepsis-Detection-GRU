# Hospital On-Premise Installation Guide

## System Requirements

### Minimum Hardware
- **CPU**: 4 cores (Intel Xeon or equivalent)
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 100 Mbps LAN

### Recommended Hardware (for 100+ patients)
- **CPU**: 8 cores
- **RAM**: 16 GB
- **Storage**: 200 GB SSD (RAID 1 for redundancy)
- **Network**: 1 Gbps LAN

### Software Requirements
- **OS**: Ubuntu 22.04 LTS or CentOS 8+ (64-bit)
- **Docker**: Version 24.0+
- **Docker Compose**: Version 2.20+

---

## Installation Steps

### Step 1: Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

### Step 2: Download Application

```bash
# Option A: Clone from GitHub
git clone https://github.com/Sympory/Sepsis-Detection-GRU.git
cd Sepsis-Detection-GRU

# Option B: Extract from provided ZIP
unzip sepsis-detection-v1.0.zip
cd sepsis-detection
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.docker .env

# Edit configuration (use nano or vi)
nano .env

# IMPORTANT: Change these values!
# - DB_PASSWORD: Strong password (min 16 characters)
# - SECRET_KEY: Random string (use: openssl rand -hex 32)
```

### Step 4: Deploy Application

```bash
# Build and start containers
docker compose up -d

# Check status
docker compose ps

# Expected output:
# NAME           STATUS         PORTS
# sepsis_db      Up (healthy)   5432/tcp
# sepsis_app     Up             0.0.0.0:5000->5000/tcp
# sepsis_nginx   Up             0.0.0.0:80->80/tcp
```

### Step 5: Verify Installation

```bash
# Check application health
curl http://localhost:5000/api/health

# Expected response:
# {"status":"healthy","model_loaded":true}

# View logs
docker compose logs -f app
```

### Step 6: Access Web Interface

Open browser and navigate to:
- **Local**: http://localhost
- **Network**: http://[server-ip-address]

Default demo credentials:
- Username: `demo`
- Password: (any password - demo mode)

---

## Post-Installation Configuration

### Enable HTTPS (Recommended)

```bash
# Generate SSL certificate (self-signed for internal use)
mkdir -p deployment/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/ssl/nginx.key \
  -out deployment/ssl/nginx.crt

# Restart nginx
docker compose restart nginx
```

### Configure Backups

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/sepsis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker exec sepsis_db pg_dump -U sepsis_user sepsis_production > \
  $BACKUP_DIR/db_backup_$DATE.sql

# Backup application data
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz data/ models/

echo "Backup completed: $DATE"
EOF

chmod +x backup.sh

# Add to cron (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

### Configure Monitoring

```bash
# View real-time resource usage
docker stats

# Set up log rotation
sudo nano /etc/logrotate.d/sepsis-app
# Add:
# /var/lib/docker/containers/*/*.log {
#   rotate 7
#   daily
#   compress
#   missingok
# }
```

---

## Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker compose down
docker compose up -d --build

# Verify
docker compose ps
```

### View Logs

```bash
# Application logs
docker compose logs -f app

# Database logs
docker compose logs -f db

# All services
docker compose logs -f
```

### Restart Services

```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart app
```

---

## Troubleshooting

### Issue: Container won't start

```bash
# Check logs
docker compose logs app

# Common fixes:
# 1. Port already in use
sudo lsof -i :5000
# Kill process or change port in docker-compose.yml

# 2. Permission issues
sudo chown -R $USER:$USER data/ models/

# 3. Memory issues
docker system prune -a  # Free up space
```

### Issue: Cannot connect to database

```bash
# Check database status
docker compose exec db psql -U sepsis_user -d sepsis_production -c "\dt"

# Reset database
docker compose down -v  # WARNING: Deletes all data
docker compose up -d
```

### Issue: Model not loading

```bash
# Verify model file exists
ls -lh models/gru_v23_best.keras

# Check permissions
chmod 644 models/gru_v23_best.keras

# Rebuild app container
docker compose up -d --build app
```

---

## Security Hardening

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Block direct access to app (force through nginx)
sudo ufw deny 5000/tcp
```

### Change Default Credentials

Edit `app.py` or `auth.py` to implement real authentication:
```python
# Replace demo mode with actual user database
# See docs/AUTHENTICATION_SETUP.md
```

### Regular Updates

```bash
# System updates (monthly)
sudo apt update && sudo apt upgrade -y

# Docker updates
sudo apt install docker-ce docker-ce-cli containerd.io

# Application updates (as released)
git pull && docker compose up -d --build
```

---

## Support

For installation assistance:
- **Email**: support@example.com
- **Phone**: +90 XXX XXX XXXX
- **Documentation**: https://github.com/Sympory/Sepsis-Detection-GRU/docs

---

## Appendix: Quick Command Reference

```bash
# Start system
docker compose up -d

# Stop system
docker compose down

# View status
docker compose ps

# View logs
docker compose logs -f

# Backup database
docker exec sepsis_db pg_dump -U sepsis_user sepsis_production > backup.sql

# Restore database
docker exec -i sepsis_db psql -U sepsis_user sepsis_production < backup.sql

# Remove everything (DANGER)
docker compose down -v
```
