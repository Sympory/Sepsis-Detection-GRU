# Hospital IT Requirements

## Overview
This document outlines the technical requirements for deploying the Sepsis Early Detection System in a hospital environment.

---

## Hardware Specifications

### Production Server (Single Hospital, up to 200 ICU Beds)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores @ 2.5 GHz | 8 cores @ 3.0 GHz |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 100 GB SSD | 500 GB SSD (RAID 1) |
| **Network** | 100 Mbps | 1 Gbps |
| **Power** | Redundant PSU | UPS + Generator backup |

### Storage Breakdown
- **Application**: ~2 GB
- **Database**: ~10-50 GB (depends on retention policy)
- **Logs**: ~5 GB/month
- **Backups**: ~50 GB (7-day retention)
- **OS + Overhead**: ~20 GB

---

## Software Requirements

### Operating System
- **Supported**: Ubuntu 22.04 LTS, CentOS 8+, Red Hat Enterprise Linux 8+
- **Architecture**: x86_64 (64-bit)
- **Kernel**: 5.x or higher

### Required Software
- **Docker Engine**: 24.0 or higher
- **Docker Compose**: 2.20 or higher
- **Python**: 3.10+ (included in Docker image)
- **PostgreSQL**: 15.x (included in Docker image)

### Optional Software
- **Git**: For version updates
- **Nginx**: For SSL termination (included in Docker)
- **Monitoring**: Prometheus, Grafana (optional)

---

## Network Requirements

### Ports
| Port | Service | Required | Purpose |
|------|---------|----------|---------|
| 80 | HTTP | Yes | Web interface |
| 443 | HTTPS | Recommended | Secure web interface |
| 22 | SSH | Yes | Remote administration |
| 5432 | PostgreSQL | No* | Internal (Docker network) |
| 5000 | Flask | No* | Internal (Docker network) |

*Not exposed to hospital network by default

### Firewall Rules
```bash
# Allow incoming
- TCP/80 (HTTP) from hospital LAN
- TCP/443 (HTTPS) from hospital LAN
- TCP/22 (SSH) from admin workstations only

# Block incoming
- All other ports

# Allow outgoing (for updates)
- TCP/80, 443 to internet (optional, for updates)
```

### Network Isolation
**Recommended**: Deploy in **DMZ** or **isolated VLAN**
- No direct internet access required
- Can operate fully offline after initial installation

---

## Security Requirements

### Data Protection
- ✅ **Encryption at Rest**: PostgreSQL data volume encrypted
- ✅ **Encryption in Transit**: HTTPS (TLS 1.2+)
- ✅ **Access Control**: Role-based authentication
- ✅ **Audit Logging**: All user actions logged

### Compliance
- **HIPAA** (US): Compliant with technical safeguards
- **KVKK** (Turkey): Personal data protection compliant
- **GDPR** (EU): Data portability and deletion supported
- **ISO 27001**: Security controls implemented

### Authentication
**Options**:
1. **Built-in**: Username/password (demo mode)
2. **LDAP/Active Directory**: Integrate with hospital AD
3. **SAML 2.0**: Single Sign-On with hospital SSO

---

## Integration Requirements

### Electronic Health Record (EHR) Integration

#### Option 1: Manual Entry
- Pre-configured for manual data input via web UI
- No IT integration required

#### Option 2: HL7 FHIR API
- RESTful API for automated data ingestion
- Supports HL7 FHIR R4 standard
- Requires custom integration (contact support)

#### Option 3: CSV Import
- Batch import patient data from CSV files
- Useful for initial deployment testing

### Example API Integration
```python
POST /api/patients/{id}/hourly-data
Content-Type: application/json

{
  "hour": 1,
  "vital_signs": {
    "HR": 85,
    "Temp": 37.2,
    "SBP": 120,
    ...
  }
}
```

---

## Backup & Recovery

### Backup Strategy
- **Frequency**: Daily automated backups
- **Retention**: 7 days (configurable)
- **Storage**: Separate backup volume or NAS
- **Content**: Database + application config

### Disaster Recovery
- **RTO** (Recovery Time Objective): < 4 hours
- **RPO** (Recovery Point Objective): 24 hours
- **Backup Location**: Separate physical server or cloud

### Backup Script (Included)
```bash
# Automated daily backup
/app/backup.sh
# Outputs to: /var/backups/sepsis/
```

---

## Performance Benchmarks

### Expected Performance (Production Hardware)
| Metric | Value |
|--------|-------|
| **Prediction Response Time** | < 500 ms |
| **Concurrent Users** | 50+ |
| **Concurrent Patients** | 200+ |
| **Database Queries/sec** | 100+ |
| **Uptime SLA** | 99.5% |

### Load Testing Results
- Tested with 100 concurrent users
- 1000 predictions/hour sustained
- Average response: 320 ms
- Peak memory: 12 GB

---

## Monitoring & Maintenance

### Health Checks
- **Application**: `/api/health` endpoint
- **Database**: PostgreSQL connection check
- **Model**: Model load verification

### Log Files
```bash
# Application logs
docker compose logs -f app

# Database logs
docker compose logs -f db

# Nginx access logs
docker compose logs -f nginx
```

### Alerts (Optional)
- High memory usage (>80%)
- Disk space low (<10GB free)
- Application errors (5xx responses)
- Database connection failures

---

## Licensing & Support

### License Type
- **On-Premise**: Perpetual license per hospital
- **Support**: 1 year included, renewable annually

### Support Channels
- **Email**: support@example.com
- **Phone**: +90 XXX XXX XXXX (business hours)
- **Remote Support**: TeamViewer or equivalent
- **Response Time**: 24 hours (standard), 4 hours (critical)

### Training
- **Initial Training**: 4-hour on-site session
- **Documentation**: Comprehensive user manuals
- **Video Tutorials**: Online training library

---

## Procurement Checklist

For hospital IT/procurement:

- [ ] Server hardware meets minimum specifications
- [ ] OS license (if using RHEL)
- [ ] UPS and backup power
- [ ] Backup storage (NAS or cloud)
- [ ] SSL certificate (if using custom domain)
- [ ] Network segmentation/VLAN configuration
- [ ] Firewall rules approved by security team
- [ ] IT staff trained on Docker administration
- [ ] Integration plan with EHR (if applicable)
- [ ] User training scheduled
- [ ] Go-live date planned

---

## Contact for Pre-Sales Consultation

Before purchasing, we offer:
- **Site Survey**: Assessment of hospital IT infrastructure
- **POC Deployment**: 30-day proof-of-concept trial
- **Integration Planning**: Custom EHR integration quotes

Contact: sales@example.com
