# 🔒 Production Güvenlik Kılavuzu

## Mevcut Güvenlik Özellikleri ✅

Sisteminizde **zaten aktif** olan güvenlikler:

1. ✅ **Bcrypt Password Hashing** - Şifreler hashli saklanıyor
2. ✅ **Session Management** - 30 dakika timeout
3. ✅ **Failed Login Tracking** - 5 deneme sonra 15 dk lock
4. ✅ **Audit Logging** - Tüm işlemler loglanıyor
5. ✅ **Role-Based Access Control** - Yetki bazlı erişim
6. ✅ **HttpOnly Cookies** - XSS koruması
7. ✅ **SameSite Strict** - CSRF koruması

---

## 🚨 Acil Production Yapılacaklar

### 1. SSL/TLS Sertifikası (ZORUNLU!)

**Neden:** Şifrelenmemiş HTTP üzerinden şifreler açık metin gidiyor!

**Çözüm - Let's Encrypt (Ücretsiz):**

```bash
# Certbot kurulumu (Windows için)
# https://certbot.eff.org/instructions

# Sertifika al
certbot certonly --standalone -d yourdomain.com
```

**Flask'da HTTPS kullan:**
```python
if __name__ == '__main__':
    app.run(
        ssl_context=('cert.pem', 'key.pem'),  # SSL sertifikaları
        host='0.0.0.0',
        port=443
    )
```

### 2. Güçlü Şifreler Zorla

**Şu anda:** Demo şifreler çok basit (`admin123`)

**Yapılacak:**
```python
# auth.py içine ekle
import re

def validate_password_strength(password):
    """Password policy enforcement"""
    if len(password) < 12:
        return False, "Şifre en az 12 karakter olmalı"
    
    if not re.search(r"[A-Z]", password):
        return False, "En az 1 büyük harf gerekli"
    
    if not re.search(r"[a-z]", password):
        return False, "En az 1 küçük harf gerekli"
    
    if not re.search(r"[0-9]", password):
        return False, "En az 1 rakam gerekli"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "En az 1 özel karakter gerekli"
    
    return True, None

# hash_password fonksiyonundan önce kontrol et
valid, error = validate_password_strength(password)
if not valid:
    raise ValueError(error)
```

### 3. Database Şifresi Güvenliği

**Şu anda:** `.env` dosyası plain text

**Yapılacak:**
```bash
# .env dosyasını .gitignore'a ekle (zaten var)
# Production'da environment variables kullan

# Windows'ta:
setx DB_PASSWORD "VeryStr0ng!P@ssw0rd123"

# Linux/Mac:
export DB_PASSWORD="VeryStr0ng!P@ssw0rd123"
```

### 4. SECRET_KEY Güvenliği

**Şu anda:** Development key kullanılıyor

**Güçlü key oluştur:**
```python
import secrets
print(secrets.token_hex(32))
# Çıktı: 3a7b9c1d2e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4z5
```

`.env` dosyasına ekle:
```
SECRET_KEY=3a7b9c1d2e4f5g6h7i8j9k0l1m2n3o4p5q6r7s8t9u0v1w2x3y4z5
```

---

## 🛡️ İleri Seviye Güvenlik

### 5. Rate Limiting (DoS Koruması)

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("5 per minute")  # Login için daha sıkı limit
def api_login():
    # ...
```

### 6. SQL Injection Koruması

**Şu anda:** Parameterized queries kullanılıyor ✅

**Ek kontrol:**
```python
# ASLA bunu YAPMAYIN:
cur.execute(f"SELECT * FROM users WHERE username = '{username}'")

# DOĞRU (zaten yapılıyor):
cur.execute("SELECT * FROM users WHERE username = %s", (username,))
```

### 7. XSS (Cross-Site Scripting) Koruması

**Frontend'de:**
```javascript
// ASLA bunu YAPMAYIN:
element.innerHTML = userInput;

// DOĞRU:
element.textContent = userInput;
// veya
element.innerText = userInput;
```

### 8. CORS Politikası Sıkılaştır

**Şu anda:** Tüm originlere açık

**Production için:**
```python
from flask_cors import CORS

CORS(app, 
     origins=['https://yourdomain.com'],  # Sadece kendi domain'iniz
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE']
)
```

### 9. Database Backup Stratejisi

**Günlük otomatik backup:**
```bash
# PostgreSQL backup script (backup.sh)
#!/bin/bash
BACKUP_DIR="/backups/postgresql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U postgres sepsis_db > "$BACKUP_DIR/sepsis_db_$TIMESTAMP.sql"

# Eski backup'ları sil (30 günden eski)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
```

**Windows Task Scheduler** ile günlük çalıştır.

### 10. Sensitive Data Encryption

**Hasta verileri encryption:**
```python
from cryptography.fernet import Fernet

# Key generate et (bir kez)
key = Fernet.generate_key()
# .env'e kaydet: ENCRYPTION_KEY=...

cipher = Fernet(os.getenv('ENCRYPTION_KEY').encode())

# Encrypt
encrypted = cipher.encrypt(sensitive_data.encode())

# Decrypt
decrypted = cipher.decrypt(encrypted).decode()
```

---

## 🔐 KVKK/GDPR Uyumluluğu

### 11. Veri Saklama Politikası

```python
# Eski verileri otomatik sil
def cleanup_old_data():
    """90 günden eski audit logları sil"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        DELETE FROM audit_logs 
        WHERE timestamp < NOW() - INTERVAL '90 days'
    """)
    
    conn.commit()
    cur.close()
    conn.close()
```

### 12. Veri İndirme ve Silme Hakkı

**API endpoint ekle:**
```python
@app.route('/api/gdpr/export', methods=['GET'])
@login_required
def export_user_data():
    """Kullanıcının tüm verilerini indir (KVKK/GDPR)"""
    # Kullanıcı verilerini JSON olarak döndür
    pass

@app.route('/api/gdpr/delete', methods=['DELETE'])
@login_required
def delete_user_account():
    """Hesap silme talebi (KVKK/GDPR)"""
    # Kullanıcı verilerini anonim hale getir
    pass
```

---

## 📊 Güvenlik Monitoring

### 13. Suspicious Activity Detection

```python
def detect_suspicious_login(user_id, ip_address):
    """Farklı lokasyondan login tespiti"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Son 24 saatteki IP'leri kontrol et
    cur.execute("""
        SELECT DISTINCT ip_address 
        FROM audit_logs 
        WHERE user_id = %s 
        AND action = 'LOGIN_SUCCESS'
        AND timestamp > NOW() - INTERVAL '24 hours'
    """, (user_id,))
    
    ips = [row[0] for row in cur.fetchall()]
    
    if ip_address not in ips and len(ips) > 0:
        # Yeni IP'den login! Email gönder
        send_security_alert(user_id, f"Yeni lokasyondan giriş: {ip_address}")
    
    cur.close()
    conn.close()
```

### 14. Log Monitoring

**Elasticsearch/Kibana veya basit log analizi:**
```python
import logging

logging.basicConfig(
    filename='/var/log/sepsis_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Kritik olayları logla
logging.warning(f"Failed login attempt: {username} from {ip_address}")
logging.error(f"SQL error: {str(e)}")
```

---

## 🚀 Production Deployment Checklist

- [ ] SSL/TLS sertifikası kuruldu
- [ ] Tüm demo şifreler değiştirildi
- [ ] SECRET_KEY production key ile değiştirildi
- [ ] DB_PASSWORD güçlü şifre
- [ ] CORS production domain ile kısıtlandı
- [ ] Rate limiting aktif
- [ ] Günlük backup yapılıyor
- [ ] .env dosyası .gitignore'da
- [ ] Firewall kuralları (sadece 443 ve 5432 portları)
- [ ] PostgreSQL remote access kısıtlı (pg_hba.conf)
- [ ] Audit logs düzenli izleniyor
- [ ] KVKK/GDPR bildirimi yapıldı
- [ ] Penetration test yapıldı
- [ ] Incident response planı var

---

## 🔬 Güvenlik Testi

### Penetration Testing Tools

```bash
# OWASP ZAP - Web güvenlik taraması
# https://www.zaproxy.org/

# SQL Injection test
sqlmap -u "http://localhost:5000/api/auth/login" --data="username=test&password=test"

# SSL/TLS test
nmap --script ssl-enum-ciphers -p 443 yourdomain.com
```

---

## 💰 Maliyetler

**Ücretsiz:**
- ✅ Let's Encrypt SSL
- ✅ PostgreSQL
- ✅ Fail2ban (attack blocking)

**Ücretli (Opsiyonel):**
- 🔐 HashiCorp Vault ($0.03/saat) - Secret management
- 📊 Datadog ($15/host/ay) - Monitoring
- 🛡️ Cloudflare Pro ($20/ay) - DDoS protection

---

## 📞 Güvenlik Breach Durumunda

1. **Immediate Response:**
   - Tüm session'ları invalid et
   - Database'i read-only'ye al
   - Yedekten restore et

2. **Investigation:**
   - Audit log'ları incele
   - Affected users'ları belirle

3. **Notification:**
   - Etkilenen kullanıcılara bildir (KVKK zorunluluğu)
   - Şifre reset'i zorunlu kıl

---

**Hazırlayan:** Gemini AI  
**Tarih:** 30 Kasım 2025  
**Revizyon:** 1.0
