# 🏥 GRU Tabanlı Erken Sepsis Tahmin Sistemi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Durum-Ara%C5%9Ft%C4%B1rma-red.svg)

**Yoğun bakım hastalarında sepsisi saatlik verilerden önceden tahmin eden derin öğrenme sistemi**

[Hızlı Başlangıç](#-hızlı-başlangıç) • [Özellikler](#-özellikler) • [Sonuçlar](#-performans) • [Güvenilirlik](#-tahmin-güvenilirliği) • [Web Uygulaması](#-web-uygulaması) • [Mimari](#️-mimari) • [Öğrenilenler](#-geliştirme-sürecinde-öğrenilenler)

</div>

---

## 📋 Genel Bakış

Bu proje, yoğun bakım ünitesindeki (YBÜ) hastaların saatlik vital bulgu ve laboratuvar verilerinden
**sepsis riskini** tahmin eden **hibrit bir yapay zekâ modeli** (yapay sinir ağı + LightGBM) ve bu modeli
kullanan, rol bazlı girişi ve normalize veritabanı olan bir **Flask web uygulaması** içerir.

Model hastanın yatışından bu yana tüm saatlik verisini zaman serisi özelliklerine (son değer, eğilim,
son ölçümden geçen süre, SIRS/SOFA benzeri skorlar…) çevirir ve sepsis olasılığını üretir.
Uygulama her tahmin için ayrıca **bu tahminin ne kadar güvenilir olduğunu** hesaplar ve gerekçeleriyle gösterir.

### 🎯 Öne Çıkanlar

- **40.336 hasta, 1,55 milyon saatlik kayıt** — PhysioNet / Computing in Cardiology Challenge 2019 verisi
- **Hasta bazlı test** — test hastalarının hiçbiri eğitimde görülmedi; test seti yalnızca bir kez kullanıldı
- **ROC-AUC 0.847, PR-AUC 0.132** — 8.066 görülmemiş test hastasında (önceki GRU: 0.806 / 0.095)
- **Hibrit model** — 281 zaman serisi özelliği üzerinde sinir ağı (MLP) + LightGBM; ağaçlar uygulamada saf NumPy ile çalışır
- **Klinik tutarlılık** — aynı saatte normal vitallere %6, septik vitallere %36 risk
- **Güvenilirlik skoru** — MC-Dropout belirsizliği, veri tamlığı, izotonik kalibrasyon
- **Rol bazlı sistem** — hemşire / doktor / yönetici, değiştirilemez denetim kaydı
- **Tek tıkla çalışır** — `baslat.bat` Python'u bulur, paketleri kontrol eder, tarayıcıyı açar

---

## 🚀 Hızlı Başlangıç

### Windows (önerilen)

1. **`baslat.bat`** dosyasına çift tıklayın.
2. ~20 saniye sonra tarayıcıda `http://localhost:5000` giriş ekranı açılır.
3. Demo hesaplarından biriyle giriş yapın (giriş ekranında tıklayınca doldurulur):

| Rol | Kullanıcı adı | Şifre |
|-----|---------------|-------|
| Yönetici | `yonetici` | `Yonetici123` |
| Doktor | `doktor` | `Doktor123` |
| Hemşire | `hemsire` | `Hemsire123` |

> Demo hesapları ilk açılışta otomatik oluşur. Gerçek kullanımda `SEPSIS_DEMO_ACCOUNTS=0` ile kapatılır,
> kullanıcılar yönetici panelinden eklenir.

Sunumdan önce her şeyin çalıştığını doğrulamak için **`sistem_kontrol.bat`**:

```
=== SepsisAI sistem kontrolü ===
  [OK] Python sürümü — 3.10
  [OK] Aktif model — v4 → models\v4\mlp.keras
  [OK] Boş form reddediliyor
  [OK] Tek vital → Yetersiz Veri
  [OK] Normal hasta tahmini — skor %2.2, Çok Düşük
  [OK] Septik hasta tahmini — skor %32.6, Çok Yüksek
  [OK] Hibrit model: iki dal da çalışıyor
  [OK] Model septik hastaya daha yüksek skor veriyor
  [OK] Hemşire alarm onaylayamaz
  [OK] Yönetici veri giremez
  [OK] Denetim kaydı silinemez
  ...
57/57 kontrol geçti.
TÜM KONTROLLER BAŞARILI — uygulama sunuma hazır.
```

Örnek hastalarla doldurmak için (stabil, kötüleşen, eksik verili):

```bash
py -3.10 scripts/demo_hastalari.py
```

### Komut satırından

```bash
py -3.10 -m pip install -r requirements.txt
py -3.10 app.py
```

> ⚠️ **Python 3.10 gerekli.** TensorFlow 2.10 daha yeni Python sürümlerinde Windows'a kurulmaz.
> Bilgisayarda birden fazla Python varsa `py` tek başına başka bir sürümü açabilir; `py -3.10` kullanın.

---

## 🌟 Özellikler

### Model

✅ **Hibrit yapay zekâ**: sinir ağı (MLP, 3 gizli katman) + LightGBM (3 seed), olasılıkların ağırlıklı ortalaması
✅ **281 zaman serisi özelliği**: son değer, ölçüm maskesi, son ölçümden geçen süre, 6/24 saatlik ve yatıştan bu yana istatistikler, değişim, SIRS/SOFA/şok indeksi
✅ **İleri doldurma**: Laboratuvar değerleri her saat ölçülmez; son ölçülen değer geçerli kabul edilir
✅ **Dengesiz veri**: Pozitif oran %1.8; sınıf ağırlığı + PR-AUC ile erken durdurma
✅ **Kalibrasyon**: İzotonik regresyon. Ham skor, gözlenen sepsis oranına çevrilir
✅ **Saf NumPy ağaçlar**: LightGBM, TensorFlow ile aynı süreçte Windows'ta çöktüğü için ağaçlar JSON'a aktarılıp NumPy ile çalıştırılır (sonuç birebir aynı)

### Web uygulaması

✅ **Hasta yönetimi**: Ekleme, silme, arama, riske göre sıralı liste
✅ **Saatlik veri girişi**: Gruplu form, canlı veri tamlığı göstergesi, alan bazında aralık uyarısı
✅ **Risk + güvenilirlik paneli**: Skor, %90 belirsizlik aralığı, güvenilirlik skoru, “Neden?” listesi
✅ **Risk trendi**: Belirsizlik bandı ve karar eşiği ile grafik
✅ **Klinik güvenlik ağı**: Hipotansiyon, laktat, ateş gibi kırmızı bayraklar riski yükseltir
✅ **Model sayfası**: Ölçülmüş metrikler, eşik seçenekleri, klinik tutarlılık testi
✅ **Açık / koyu tema**, mobil uyumlu

---

## 📊 Performans

Tüm değerler, eğitimde **hiç görülmemiş 8.066 test hastası** (308.743 saatlik kayıt, %1.8 pozitif) üzerindedir.
Model seçimi, birleşim ağırlığı, kalibrasyon ve eşik **validasyon** hastalarıyla yapıldı; test seti **bir kez** kullanıldı.

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **Hibrit v4 (aktif)** — 0.7 × LightGBM + 0.3 × sinir ağı | **0.847** | **0.132** |
| └ LightGBM dalı | 0.847 | 0.131 |
| └ Sinir ağı dalı (MLP) | 0.823 | 0.113 |
| GRU v3 (önceki model, son 6 saat) | 0.806 | 0.095 |

| Metrik | Değer |
|--------|-------|
| ROC-AUC (YBÜ'de ≥ 6 saat geçmiş hastalar) | 0.858 |
| Brier (ham → kalibre) | 0.023 → 0.017 |
| ECE (ham → kalibre) | 0.048 → 0.002 |

### 5 katlı çapraz doğrulama (40.336 hasta, hasta bazlı)

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **Hibrit** | **0.842 ± 0.009** | **0.129 ± 0.010** |
| LightGBM | 0.840 ± 0.010 | 0.128 ± 0.009 |
| Sinir ağı (MLP) | 0.820 ± 0.006 | 0.112 ± 0.007 |

Tek bölmedeki test sonucu (0.847) bu aralıkla tutarlı; sonuç tek bir veri bölmesine bağlı değil (`scripts/cv_hybrid.py`).

### Klinik skorlarla karşılaştırma ve PhysioNet Utility (test hastaları)

| Yöntem | ROC-AUC | PR-AUC | Utility |
|--------|---------|--------|---------|
| **Hibrit v4** | **0.847** | **0.132** | **0.427** |
| GRU v3 | 0.806 | 0.095 | 0.358 |
| NEWS2 (kısmi, ≥ 5) | 0.682 | — | 0.115 |
| SIRS (≥ 2) | 0.649 | — | 0.054 |
| qSOFA (kısmi, ≥ 2) | 0.593 | — | 0.023 |

**Utility**, PhysioNet 2019'un resmi metriğidir: sepsisten 12–6 saat önceki alarmı ödüllendirir, geç ve yanlış alarmı
cezalandırır (1 = mükemmel, 0 = hiç alarm vermemek). Model eşiği validasyonda seçildi. Yarışma kazananı 0.36 aldı ama
o skor **görülmemiş üçüncü bir hastanede** ölçüldü; bizimki aynı iki hastanenin ayrılmış test hastalarında, bu yüzden
**doğrudan karşılaştırılamaz**. qSOFA ve NEWS2 bilinç (GKS) verisi olmadığı için kısmi hesaplandı (`scripts/evaluate_clinical.py`).

### İki çalışma noktası

| Eşik | Duyarlılık | Özgüllük | PPV | NPV | Kullanım |
|------|-----------|----------|-----|-----|----------|
| **0.33** (varsayılan, F1) | %30.7 | %96.8 | %15.0 | %98.7 | Az alarm |
| **0.066** (tarama) | %80.6 | %74.5 | %5.5 | %99.5 | Az vaka kaçar |

Tarama eşiğinde önceki modele göre **özgüllük %63.9 → %74.5**: aynı duyarlılıkta yanlış alarmlar yaklaşık %30 azaldı.

### Gerçek hasta akışında (uygulamanın tam karar mantığı)

Test hastaları saat saat uygulamanın kendi kurallarından (model + klinik kırmızı bayraklar + veri kapısı) geçirildi
(`scripts/hasta_bazli_test.py`; 425 sepsis hastası, 7.480 sepsis olmayan):

| Karar kuralı | Klinik tanıdan önce alarm alan sepsis hastası | En az bir alarm alan sepsis olmayan hasta |
|---|---|---|
| “Yüksek” ve üstü | %63 (önceki %64) | **%13** (önceki %16) |
| Tarama eşiği | %94 (önceki %99) | **%58** (önceki %89) |

### Klinik tutarlılık testi

Normal vitaller (HR 78, Ateş 36.8, SKB 125) ile son 6 saatte septik tabloya (HR 130, Ateş 38.9, SKB 82,
SpO2 88, Laktat 4.5, WBC 17) dönen hasta:

| YBÜ saati | Normal | Septik |
|-----------|--------|--------|
| 6 | %3.9 | **%35.5** |
| 24 | %6.0 | **%35.6** |
| 72 | %10.3 | **%65.3** |

### Tahmin açıklaması

Her tahminde **skoru en çok artıran ve azaltan faktörler** gösterilir (ör. “Ateş 39.1 °C, 6 saatte +2.3 → +15 puan”).
LightGBM ağaçlarındaki karar yolu izlenir; her bölünmede skorun değişimi o özelliğe yazılır (Saabas yöntemi) ve
281 özellik klinik değişkene göre toplanır (`explain.py`). Katkıların toplamı modelin skoruna eşittir (testle doğrulanır).

### En önemli özellikler

YBÜ'de geçen süre, son FiO2 ölçümünden bu yana geçen süre, SpO2/nabız ölçüm sayısı, son laktat ölçümünden
bu yana geçen süre, 6 saatlik en yüksek ateş, şok indeksi. **Not:** “son ölçümden geçen süre” ve “ölçüm sayısı”
özellikleri hekimin davranışını da yansıtır (şüphelenince test ister); test isteme alışkanlığı farklı bir
hastanede performans değişebilir.

Grafikler: `models/v4/evaluation.png` (ROC, PR, güvenilirlik diyagramı).

---

## 🛡️ Tahmin Güvenilirliği

İlk sürümde boş form gönderildiğinde model **“düşük risk”** diyordu. Nedeni: eksik alanlar eğitim
verisinin medyanıyla (ortalama, sağlıklı bir hasta) dolduruluyordu. Artık:

| Kontrol | Ne yapar |
|---------|----------|
| **Minimum veri kapısı** | HR, SpO2, Ateş, Solunum, MAP'ten en az 3'ü yoksa risk verilmez → **“Yetersiz Veri”** |
| **Veri tamlığı** (%45) | Gerçekten girilen değerlerin ağırlıklı oranı (temel vitaller %70, kritik lab %20) |
| **Zaman kapsamı** (%20) | 6 saatlik pencerenin kaçı gerçek ölçüm |
| **Model kararlılığı** (%25) | **MC-Dropout**: Dropout açık 30 tahmin. Kaçı aynı kararı veriyor |
| **Dağılım uyumu** (%10) | Eğitim verisinde çok nadir görülen değerler |
| **Model–klinik çelişkisi** | Kırmızı bayraklar varken model düşük derse uyarı verilir, güvenilirlik düşürülür |

Sonuç 0–100 arası bir skor ve **Yüksek / Orta / Düşük / Yetersiz Veri** seviyesidir.
Test setinde model kararlı olduğunda doğruluk %61, kararsız olduğunda %45'tir.

---

## 👥 Kullanıcılar, Roller ve Denetim

Her istek oturum ve rol kontrolünden geçer. Rol bazlı yetkiler `security.py → PERMISSIONS` içinde tek yerde tanımlıdır.

| İşlem | Hemşire | Doktor | Yönetici |
|-------|:------:|:------:|:-------:|
| Hasta listesi, risk, model bilgisi | ✓ | ✓ | ✓ |
| Hasta kabul, saatlik vital/lab girişi | ✓ | ✓ | — |
| Alarm onayı (klinik not zorunlu) | — | ✓ | — |
| Taburcu / yeniden kabul (arşiv), sepsis sonucunu işaretleme | — | ✓ | — |
| Kalıcı silme | — | — | ✓ |
| Kullanıcı ekleme, rol değiştirme, kilit açma, şifre sıfırlama | — | — | ✓ |
| Tüm denetim kayıtları, veritabanı paneli, veri seti dışa aktarma | — | — | ✓ |
| Hastanın kayıt geçmişi (kim, ne zaman, ne girdi) | ✓ | ✓ | ✓ |

**Güvenlik:**
- Şifreler tuzlu **scrypt** hash olarak saklanır; şifre kuralı en az 8 karakter, harf ve rakam.
- Oturum token'ı **HttpOnly, SameSite=Strict** çerezde tutulur, veritabanında yalnızca SHA-256 özeti saklanır.
- **30 dakika** işlem yapılmazsa oturum kapanır; **5 hatalı** girişte hesap **15 dakika** kilitlenir.
- Proje klasöründeki dosyalar (veritabanı, `.env`, modeller) tarayıcıdan erişilemez.

**Denetim kaydı (`audit_logs`):** giriş/çıkış, başarısız giriş, hesap kilidi, yetkisiz deneme, hasta kabul/taburcu/silme,
veri girişi (üzerine yazmada eski ve yeni değerler), alarm açılması ve onayı, kullanıcı işlemleri. Her kayıtta zaman,
kullanıcı, rol, IP ve ilgili hasta tutulur. Kayıtlar **SQL tetikleyicileriyle değiştirilemez ve silinemez**.

## 🗄️ Veritabanı (SQLite, normalize)

```
users ──< sessions
  │
  ├──< audit_logs                (UPDATE/DELETE tetikleyiciyle engelli)
  │
patients ──< observations ──< measurements     (her ölçüm ayrı satır)
   │              │
   │              └──── predictions            (1–1: skor, risk, güvenilirlik, model sürümü)
   ├──< alarms                                 (onaylayan, zaman, klinik not)
   └──── outcomes                              (1–1: doktorun işaretlediği sepsis sonucu)
```

| Tablo | İçerik | Kısıtlar |
|-------|--------|----------|
| `users` | Hesaplar | `username` UNIQUE, `role` CHECK (admin/doctor/nurse) |
| `sessions` | Açık oturumlar | token SHA-256, `user_id` → users (CASCADE) |
| `patients` | Hastalar | `patient_id` UNIQUE, `age` CHECK 0–120, `status` CHECK (active/discharged) |
| `observations` | Saatlik kayıt | UNIQUE(hasta, saat), `entered_by` → users |
| `measurements` | Ölçüm değerleri | PK(kayıt, değişken) |
| `predictions` | Model çıktısı | kayıt başına bir tahmin, model sürümüyle |
| `alarms` | Yüksek risk uyarıları | `status` CHECK (open/acknowledged), `acknowledged_by` → users |
| `outcomes` | Gerçek sonuç (veri seti etiketi) | PK = FK → patients (1–1), `sepsis` CHECK (yes/no/unknown) |
| `audit_logs` | Denetim kaydı | değiştirilemez |

- Yabancı anahtarlar açık: hasta kalıcı silinince ölçümleri, tahminleri ve alarmları CASCADE ile silinir; denetim kaydı kalır.
- Eski sürümün veritabanı (`hourly_data`, ölçümler JSON metin) ilk açılışta **otomatik yedeklenip** yeni şemaya taşınır.
- Yönetici panelindeki **Veritabanı** sayfası tabloları, kayıt sayılarını ve ilişkileri gösterir.

### Kullanımdan veri seti toplama

Doktor hastayı taburcu ederken **gerçek sonucu** işaretler: sepsis tanısı kondu mu (Evet / Hayır / Belirsiz) ve evetse
klinik başlangıç saati. Sonuç sonradan düzeltilebilir (ör. kültür sonucu geç geldiyse). Böylece sistem kullanıldıkça
etiketli yerel veri birikir.

Yönetici **Veritabanı → CSV indir** ile (ya da `py -3.10 scripts/veri_seti_disa_aktar.py`) bu veriyi dışa aktarır:
- Yalnızca taburcu edilmiş ve sonucu Evet/Hayır işaretlenmiş hastalar
- **Kimlik bilgisi yok:** ad ve protokol no yerine `L00001` gibi sıra numarası
- **PhysioNet 2019 formatı**, aynı etiket kuralı (klinik başlangıçtan 6 saat önce `SepsisLabel = 1`)

Dosya `data/train.csv` ile birleştirilip istenirse model yeniden eğitilebilir; yeniden eğitim otomatik yapılmaz.

## 🌐 Web Uygulaması

### API

Tüm `/api/*` uçları (giriş ve sağlık kontrolü hariç) oturum ister; parantez içi gereken izindir.

| Metot | Uç nokta | Açıklama |
|-------|----------|----------|
| GET | `/`, `/login` | Arayüz (oturum yoksa giriş ekranına yönlendirir) |
| POST | `/api/auth/login`, `/api/auth/logout` | Giriş / çıkış |
| GET | `/api/auth/me` | Oturumdaki kullanıcı ve izinleri |
| POST | `/api/auth/change-password` | Kendi şifresini değiştirme |
| GET | `/api/patients?status=active\|discharged` | Hastalar, son risk, açık alarm sayısı |
| POST | `/api/patients` | Hasta kabul (hemşire, doktor) |
| GET | `/api/patients/<id>` | Hasta detayı, saatlik kayıtlar, açık alarm |
| POST | `/api/patients/<id>/hourly-data` | Saatlik veri → tahmin + güvenilirlik (+ alarm) |
| POST | `/api/patients/<id>/discharge`, `/readmit` | Taburcu (sonuç zorunlu) / yeniden kabul (doktor) |
| DELETE | `/api/patients/<id>` | Kalıcı silme (yönetici) |
| GET | `/api/patients/<id>/history` | Hastanın denetim geçmişi |
| GET | `/api/alarms?status=open\|acknowledged` | Alarmlar |
| POST | `/api/alarms/<id>/acknowledge` | Alarm onayı + klinik not (doktor) |
| GET/POST/PATCH | `/api/users` | Kullanıcı yönetimi (yönetici) |
| GET | `/api/audit-logs` | Denetim kayıtları, filtreli (yönetici) |
| PUT | `/api/patients/<id>/outcome` | Sepsis sonucunu düzelt (doktor) |
| GET | `/api/admin/db-stats` | Tablo kayıt sayıları, veri seti özeti (yönetici) |
| GET | `/api/admin/dataset.csv` | Anonim, etiketli veri seti (yönetici) |
| GET | `/api/model-info` | Aktif model, metrikler, kalibrasyon istatistikleri |
| GET | `/api/health` | Sistem durumu |

Örnek yanıt (demo hasta “Mehmet Kaya”, YBÜ 24. saat, hibrit model):

```json
{
  "prediction": 0.421,
  "risk_level": "Çok Yüksek",
  "is_sepsis_risk": true,
  "threshold": 0.33,
  "reliability": {
    "score": 96.0,
    "level": "Yüksek",
    "components": {"completeness": 91.2, "temporal": 100.0, "certainty": 100.0, "in_distribution": 100.0},
    "mc_dropout": {"mean": 0.419, "p05": 0.388, "p95": 0.462, "agreement": 1.0, "n": 30},
    "components_model": {"lightgbm": 0.407, "neural_net": 0.454, "weight_lightgbm": 0.7},
    "calibrated_probability": 0.137,
    "reasons": ["MAP, SBP/DBP'den hesaplandı (60.2 mmHg)"]
  }
}
```

### Risk seviyeleri (`config.py`)

| Skor | Seviye | Öneri |
|------|--------|-------|
| < 0.10 | 🟢 Çok Düşük | Rutin takip |
| 0.10 – 0.20 | 🔵 Düşük | Takip |
| 0.20 – 0.35 | 🟠 Orta | Yakın takip, ek lab |
| 0.35 – 0.55 | 🔴 Yüksek | Klinik değerlendirme |
| ≥ 0.55 | 🚨 Çok Yüksek | Sepsis protokolü |
| — | ⚪ Yetersiz Veri | En az 3 temel vital girin |

---

## 🧪 Testler

```bash
py -3.10 -m pytest                    # tüm testler (model dahil)
py -3.10 -m pytest -m "not model"     # modelsiz hızlı testler
sistem_kontrol.bat                    # sunum öncesi uçtan uca kontrol
```

| Dosya | Ne test ediliyor |
|-------|------------------|
| `tests/test_security.py` | Şifre hash'i ve kuralı, 5 hatada kilit, 30 dk oturum zaman aşımı, token'ın hash'li saklanması, rol × uç nokta yetki matrisi |
| `tests/test_database.py` | CHECK / UNIQUE / FK kısıtları, CASCADE silme, değiştirilemez denetim kaydı, eski veritabanından taşıma |
| `tests/test_features.py` | **Geleceğe bakmama** (sonraki saatler eklenince geçmiş özellikler değişmez), ileri doldurma, hastalar arası karışmama |
| `tests/test_gbm_numpy.py` | NumPy ağaç değerlendiricisinde eksik değer kuralları, açıklama katkılarının toplamı = skor |
| `tests/test_utility.py` | Vektörel Utility hesabının resmi tanımla birebir aynı olması |
| `tests/test_api.py` | Boş form, yetersiz veri, septik > normal, alarm, açıklama, taburcu, anonim veri seti |

## 🏗️ Mimari

### Sistem

```
Tarayıcı (index.html, script.js, style.css)
        │  JSON / REST
        ▼
Flask API (app.py)
        │
        ├── security.py      → giriş, oturum, rol bazlı yetki
        ├── db.py            → SQLite şeması, taşıma, sorgular, denetim kaydı ── patients.db
        └── ml_service.py    → tahmin
              ├── features.py     → ileri doldurma, ölçüm maskeleri, ölçekleme
              ├── GRU modeli      → models/v3/model.keras
              └── reliability.py  → MC-Dropout, tamlık, kalibrasyon, güvenilirlik skoru
```

### Model (v4 — hibrit)

```
Hastanın yatıştan bu yana saatlik verisi (girilmeyen saatler boş satır)
        │
        ▼
features_v4.py → 281 özellik (son saat için)
        │
        ├──────────────────────────────┐
        ▼                              ▼
LightGBM (3 seed, 556 ağaç)       Sinir ağı (MLP, 114.945 ağırlık)
saf NumPy ile                     GaussianNoise → Dense 256 → BN → Dropout 0.5
                                  → Dense 128 → BN → Dropout 0.5
                                  → Dense 64 → Dropout 0.5 → Sigmoid
        │                              │
        └──────── 0.7 × + 0.3 × ───────┘
                       │
                       ▼
          İzotonik kalibrasyon → P(Sepsis)
          MC-Dropout (sinir ağı dalı) → %90 belirsizlik aralığı
```

Önceki model (GRU v3, son 6 saat × 71 özellik) `SEPSIS_MODEL=v3` ile hâlâ seçilebilir;
katman katman açıklaması [docs/MIMARI_ACIKLAMA.md](docs/MIMARI_ACIKLAMA.md).

---

## 🔬 Yöntem

### Veri hazırlama (`features.py`)

1. **Temizlik**: Fizyolojik olarak imkânsız değerler silinir (ör. HR > 300, Ateş < 25)
2. **Ölçüm maskesi**: Her değişken için “bu saat ölçüldü mü” (0/1)
3. **İleri doldurma**: Hasta içinde son ölçülen değer taşınır
4. **Medyan doldurma + standartlaştırma**: Yalnızca **eğitim hastalarıyla** fit edilir
5. **Kayan pencere**: Her saat için son 6 saat; ilk saatlerde ilk ölçüm geriye kopyalanır
6. **Bölme**: Hasta bazlı, sepsis durumuna göre tabakalı %60 / %20 / %20

### Hibrit model eğitimi (`scripts/train_hybrid.py`)

| Adım | Ayrıntı |
|------|---------|
| LightGBM | 31 yaprak, en az 1000 örnek/yaprak, L2 = 20, özellik oranı 0.4, öğrenme hızı 0.03, 3 seed; val PR-AUC ile erken durdurma |
| Sinir ağı | Adam (lr 1e-3), batch 4096, pozitif sınıf ağırlığı 5, L2 1e-4, Dropout 0.5; val PR-AUC ile erken durdurma |
| Birleşim | Ağırlık validasyonda PR-AUC'ye göre seçildi (LightGBM 0.7, sinir ağı 0.3) |
| Kalibrasyon ve eşik | İzotonik regresyon, F1 ve %80 duyarlılık eşikleri validasyonda |
| Test | Yalnızca bir kez, en sonda |
| Süre | ~8 dk (RTX 4060 + işlemci) |

```bash
py -3.10 scripts/train_hybrid.py                 # hibrit model → models/v4/
py -3.10 scripts/hasta_bazli_test.py             # uygulama mantığıyla hasta bazlı test
```

Optimizasyon deneyleri (`scripts/experiments/`, hepsi validasyonda):

| Deney | val ROC-AUC | val PR-AUC |
|-------|-------------|------------|
| GRU v3 (6 saat) | 0.804 | 0.097 |
| GRU, 24 saatlik pencere | 0.813 | 0.101 |
| GRU + en önemli 80 zaman serisi özelliği | 0.819 | 0.099 |
| MLP + zaman serisi özellikleri | 0.829 | 0.120 |
| LightGBM + zaman serisi özellikleri (257) | 0.847 | 0.130 |
| LightGBM + yatıştan bu yana özellikler (281) | 0.850 | 0.133 |
| **Hibrit (LightGBM 0.7 + MLP 0.3)** | **0.849** | **0.133** |

**Bulgu:** Kazancın çoğu model türünden değil **özellik mühendisliğinden** geldi. Aynı özelliklerle sinir ağı da
0.804'ten 0.829'a çıktı. Tekrarlayan katmanlı GRU'lar bu veride 1–2 epoch sonra hastaları ezberlemeye başlıyor;
özellikler geçmişi zaten özetlediği için düz bir sinir ağı daha iyi genelliyor.

### GRU v3 eğitimi (`scripts/train_v3.py`)

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam (lr = 5e-4, clipnorm = 1) |
| Kayıp | Binary Cross-Entropy, pozitif sınıf ağırlığı 5 |
| Batch | 2048 |
| Erken durdurma | val PR-AUC, sabır 3, en iyi ağırlıklar geri yüklenir |
| LR azaltma | val PR-AUC, faktör 0.5, sabır 2 |
| Donanım | NVIDIA RTX 4060 Laptop, ~3 dk |

```bash
py -3.10 scripts/train_v3.py --epochs 25        # model + kalibrasyon + testler → models/v3/
py -3.10 scripts/baseline_lightgbm.py           # aynı veride LightGBM karşılaştırması
```

---

## 💡 Geliştirme Sürecinde Öğrenilenler

| Sürüm | ROC-AUC | Sorun | Çözüm |
|-------|---------|-------|-------|
| **v23** | 0.880 | Veri **saat saat rastgele** bölündü. Aynı hastanın saatleri hem eğitimde hem testte vardı. Yaş + cinsiyet + yatış süresi hastaların %93'ünü tek başına tanımladığı için model hastayı “tanıyordu”. | Hasta bazlı bölme |
| **v24 (56 özellik)** | 0.9997 | PCT, CRP, Presepsin gibi 22 biyobelirteç **sentetikti** ve etiketi ele veriyordu. | Bu sütunlar modelden çıkarıldı |
| **Optimized v1** | 0.809 | Eksikler medyanla dolduruldu. Labların %90+'ı eksik olduğu için model değeri değil “ölçülmüş mü” bilgisini öğrendi: **düşük laktat riski artırıyordu**, septik hastaya %7 veriyordu. | İleri doldurma + açık ölçüm maskeleri |
| **v3 (ilk deneme)** | 0.808 | Eğitim AUC 0.92, validasyon 0.75: `HospAdmTime` ile hasta ezberleniyordu. | `HospAdmTime` çıkarıldı, gürültü + L2 eklendi |
| **v3** | 0.806 | Model son 6 saatin ham değerlerini görüyor; eğilimi kendisi çıkarmak zorunda | Zaman serisi özellikleri |
| **v4 (hibrit)** | **0.847** | — | Özellik mühendisliği + sinir ağı/LightGBM birleşimi, saf NumPy ağaçlar |

**Ana ders:** Daha yüksek AUC her zaman daha iyi model demek değildir. 0.88'lik model, görmediği
hastada 0.81'lik modelden daha iyi değildi. Doğru değerlendirme yöntemi metrik kadar önemlidir.

---

## 🗺️ Yol Haritası (6 hafta)

İlk 6 haftalık planın **özellik mühendisliği, uzun pencere ve topluluk (ensemble)** adımları yapıldı:
ROC-AUC 0.806 → **0.847** (hedef 0.83–0.85). Kalan adımlar:

| Hafta | Konu | Ana teknikler |
|-------|------|---------------|
| ✅ 1 | Değerlendirme altyapısı | 5 katlı GroupKFold, PhysioNet Utility, SIRS/qSOFA/NEWS2 karşılaştırması — **yapıldı** |
| ✅ 2 | Özellik mühendisliği | Ölçümden geçen süre, 6/24 saatlik ve yatıştan bu yana trendler, SOFA, şok indeksi — **yapıldı** |
| ◐ 3 | Mimari | 24 saatlik pencere GRU denendi — **yapıldı**; GRU-D, TCN, Transformer, Optuna kaldı |
| ✅ 4 | Topluluk | LightGBM (3 seed) + sinir ağı birleşimi, kalibrasyon — **yapıldı** |
| ◐ 5 | Açıklanabilirlik | Tahmin başına faktör açıklaması (Saabas) — **yapıldı**; TreeSHAP, conformal prediction kaldı |
| ◐ 6 | Dış doğrulama ve rapor | pytest (63 test) — **yapıldı**; hastane A → B testi, alt grup analizi, Docker kaldı |

**Hedef:** ROC-AUC 0.83–0.85 ✅ (0.847; CV 0.842 ± 0.009) · Utility ≥ 0.30 ✅ (0.427, iç test).

Ayrıntılar: [docs/PROJE_PLANI_6_HAFTA.md](docs/PROJE_PLANI_6_HAFTA.md) · Mimari açıklaması: [docs/MIMARI_ACIKLAMA.md](docs/MIMARI_ACIKLAMA.md)

---

## 📁 Proje Yapısı

```
├── baslat.bat                  # Tek tıkla başlatma (Windows)
├── sistem_kontrol.bat / .py    # Uçtan uca test
├── app.py                      # Flask API rotaları
├── security.py                 # Giriş, oturum, roller ve izinler
├── db.py                       # SQLite şeması, taşıma, sorgular, denetim kaydı
├── ml_service.py               # Model yükleme ve tahmin
├── config.py                   # Model seçimi, eşikler, risk seviyeleri
├── features.py                 # v3 özellikleri (ileri doldurma + ölçüm maskeleri)
├── features_v4.py              # v4 zaman serisi özellikleri (eğitim ve uygulama ortak)
├── gbm_numpy.py                # LightGBM ağaçlarını saf NumPy ile çalıştırır
├── reliability.py              # Güvenilirlik skoru, MC-Dropout
├── explain.py                  # Tahmin açıklaması: skoru en çok etkileyen faktörler
├── index.html / script.js / style.css   # Uygulama arayüzü
├── login.html / login.js       # Giriş ekranı
│
├── scripts/
│   ├── train_hybrid.py         # v4 hibrit model: eğitim, kalibrasyon, tek seferlik test
│   ├── hasta_bazli_test.py     # Uygulama karar mantığıyla hasta bazlı test
│   ├── evaluate_clinical.py    # SIRS / qSOFA / NEWS2 karşılaştırması + PhysioNet Utility
│   ├── cv_hybrid.py            # 5 katlı hasta bazlı çapraz doğrulama
│   ├── enrich_trees.py         # Ağaçlara açıklama için düğüm değerlerini ekler (model değişmeden)
│   ├── experiments/            # Optimizasyon deneyleri (GRU pencere, MLP, LightGBM ayarı)
│   ├── train_v3.py             # v3 eğitimi + kalibrasyon + değerlendirme
│   ├── baseline_lightgbm.py    # Karşılaştırma modeli
│   ├── fit_calibration.py      # Eski model için kalibrasyon
│   ├── demo_hastalari.py       # Sunum için örnek hastalar
│   └── veri_seti_disa_aktar.py # Uygulamada biriken etiketli veriyi CSV'ye aktarır
│
├── models/
│   ├── v4/                     # Aktif hibrit model: ağaçlar (JSON), sinir ağı, ölçekleyici, kalibratör
│   ├── v3/                     # Önceki GRU modeli (SEPSIS_MODEL=v3)
│   └── optimized/              # Eski model (SEPSIS_MODEL=legacy ile seçilir)
│
├── tests/                      # pytest birim ve entegrasyon testleri
├── legacy/                     # Artık kullanılmayan eski dosyalar (açıklaması içinde)
├── data/train.csv              # PhysioNet 2019 (1,55M satır, 40.336 hasta)
└── docs/
    ├── MIMARI_ACIKLAMA.md      # Katman katman model açıklaması
    ├── PROJE_PLANI_6_HAFTA.md  # Deney geçmişi, literatür, 6 haftalık plan
    ├── SUNUM_REHBERI.md        # Sunum akışı ve olası sorular
    └── PROJE_DEGERLENDIRME.md  # Kod incelemesi ve yapılacaklar
```

---

## 📚 Kaynaklar

**Veri seti**
- Reyna MA, et al. (2020). “Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.” *Critical Care Medicine* 48(2):210–217.

**Klinik**
1. Singer M, et al. (2016). “The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).” *JAMA*.
2. Surviving Sepsis Campaign Guidelines (2021).

**Teknik**
1. Cho K, et al. (2014). “Learning Phrase Representations using RNN Encoder-Decoder.” *EMNLP*.
2. Gal Y, Ghahramani Z. (2016). “Dropout as a Bayesian Approximation.” *ICML*. (MC-Dropout)
3. Che Z, et al. (2018). “Recurrent Neural Networks for Multivariate Time Series with Missing Values.” *Scientific Reports*. (Ölçüm maskeleri)

---

## ⚠️ Uyarı

**Bu yazılım yalnızca ARAŞTIRMA ve EĞİTİM amaçlıdır.**

- ❌ Klinik kullanım için onaylı değildir
- ❌ Hekim değerlendirmesinin yerini tutmaz
- ❌ Varsayılan eşikte sepsis vakalarının çoğunu kaçırır. “Düşük risk” sepsis olmadığı anlamına gelmez

---

## 📄 Lisans

MIT Lisansı. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

**Ahmet Sami Ceyhan** — Python Programlama / Yapay Sinir Ağları Projesi

---

<div align="center">

Daha iyi sağlık sonuçları için ❤️ ile geliştirildi

</div>
