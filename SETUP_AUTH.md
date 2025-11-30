# Faz 1: Authentication & Multi-Tenant Sistemi - Kullanma Kılavuzu

## 📋 Kurulum Adımları

### 1. Gerekli Paketleri Yükleyin

```bash
pip install psycopg2-binary bcrypt python-dotenv
```

### 2. PostgreSQL Kurulumu

PostgreSQL'i sisteminize kurun:
- **Windows**: https://www.postgresql.org/download/windows/
- Kurulum sırasında şifre belirleyin (örn: `postgres`)

### 3. Veritabanı Oluşturun

PostgreSQL'e bağlanın ve veritabanı oluşturun:

```sql
CREATE DATABASE sepsis_db;
```

### 4. Environment Dosyasını Yapılandırın

`.env.example` dosyasını `.env` olarak kopyalayın ve düzenleyin:

```bash
cp .env.example .env
```

`.env` dosyasında şifrenizi güncelleyin:
```
DB_PASSWORD=your_actual_postgres_password
```

### 5. Veritabanını Başlatın

```bash
cd database
python init_db.py
```

Onay sorusuna `yes` yazın.

## 🔑 Demo Giriş Bilgileri

Veritabanı başlatıldıktan sonra şu kullanıcılarla giriş yapabilirsiniz:

### System Admin
- **Username**: `admin`
- **Password**: `admin123`
- **Hospital**: Demo Hastane (Test)

### Doctor
- **Username**: `demo_doctor`
- **Password**: `doctor123`
- **Hospital**: Demo Hastane (Test)

### Nurse
- **Username**: `demo_nurse`
- **Password**: `nurse123`
- **Hospital**: Demo Hastane (Test)

## 🚀 Uygulamayı Çalıştırma

### 1. Flask App'i Güncelleyin

`app.py` dosyasının başına şunları ekleyin:

```python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

# Import auth functions
from auth import login_required, require_role, g

# Import auth endpoints
import app_auth_endpoints
```

### 2. Uygulamayı Başlatın

```bash
python app.py
```

### 3. Login Sayfasını Açın

Tarayıcınızda: `http://localhost:5000/login.html`

## 📊 Özellikler

### ✅ Tamamlanan
1. **Multi-Tenancy** - Hastane bazlı veri izolasyonu
2. **Role-Based Access Control** - 5 rol: admin, hospital_admin, doctor, nurse, viewer
3. **Secure Authentication** - Bcrypt password hashing
4. **Session Management** - 30 dakika timeout, auto-extend
5. **Failed Login Tracking** - 5 deneme sonra 15 dakika lock
6. **Audit Logging** - Tüm kullanıcı işlemleri loglanır
7. **Modern Login UI** - Responsive, professional design

### 🎯 API Endpoints

#### Authentication
- `GET /api/hospitals` - Hastane listesi
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Mevcut kullanıcı bilgisi
- `POST /api/auth/change-password` - Şifre değiştirme

#### User Management (Admin only)
- `GET /api/users` - Kullanıcı listesi
- `GET /api/audit-logs` - Audit log görüntüleme

## 🔒 Güvenlik Notları

⚠️ **ÖNEMLİ**: Production ortamında mutlaka yapılmalıdır:

1. **Şifreleri değiştirin** - Tüm demo şifrelerini değiştirin
2. **SECRET_KEY güvenliği** - Güçlü bir secret key kullanın
3. **HTTPS kullanın** - SSL sertifikası ekleyin
4. **Database şifresi** - Güçlü database şifresi belirleyin
5. **Firewall** - PostgreSQL portunu sadece gerekli IP'lere açın

## 🐛 Sorun Giderme

### Problem: "psycopg2 module not found"
```bash
pip install psycopg2-binary
```

### Problem: "Connection refused" (PostgreSQL)
- PostgreSQL service'inin çalıştığından emin olun
- Port 5432'nin açık olduğunu kontrol edin
- `pg_hba.conf` dosyasında localhost erişimine izin verildiğinden emin olun

### Problem: "Authentication failed"
- `.env` dosyasındaki şifrenin doğru olduğundan emin olun
- PostgreSQL kullanıcı şifresini kontrol edin

## 📝 Sırada Ne Var?

**Faz 2**: Yeni Klinik Biomarker'lar
- 22 yeni özellik ekleme (PCT, CRP, IL-6, etc.)
- Veri giriş formlarını güncelleme
- Model re-training

---

**Hazırlayan**: Gemini AI  
**Tarih**: 30 Kasım 2025  
**Faz**: 1 - Authentication & Infrastructure
