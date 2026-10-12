# Veritabanı Rehberi — Bitirme Ödevi Sunumuna Hazırlık

Bu belge SepsisAI veritabanını, temel kavramları **kendi projenizdeki örneklerle** anlatır.
Kod: `db.py` (şema, taşıma, sorgular), `security.py` (şifre ve oturum).

---

## 1. Genel resim

**İlişkisel veritabanı** (relational database): veriler tablolarda tutulur, tablolar birbirine **anahtarlarla** bağlanır,
sorgular **SQL** ile yazılır. Projede **SQLite** kullanılıyor: sunucu kurulumu gerektirmeyen, tek dosyalık (`patients.db`)
bir ilişkisel veritabanı. Python'da `sqlite3` modülüyle gelir.

```
users ──< sessions
  │
  ├──< audit_logs                 (değiştirilemez)
  │
patients ──< observations ──< measurements
   │              │
   │              └──── predictions    (1–1)
   ├──< alarms
   └──── outcomes       (1–1)
```

`──<` işareti **bire-çok** (one-to-many) ilişkidir: bir hastanın çok saatlik kaydı olur, her kaydın tek bir hastası vardır.

| Tablo | Ne tutar | Bir satır = |
|-------|----------|-------------|
| `users` | Hesaplar (yönetici, doktor, hemşire) | bir kullanıcı |
| `sessions` | Açık oturumlar | bir giriş |
| `patients` | Hastalar | bir hasta |
| `observations` | Saatlik kayıt başlığı (hangi hasta, kaçıncı saat, kim girdi) | bir hasta-saat |
| `measurements` | Ölçüm değerleri | bir değer (ör. nabız = 118) |
| `predictions` | Model sonucu (skor, risk, güvenilirlik, model sürümü) | bir tahmin |
| `alarms` | Yüksek risk uyarıları ve doktor onayı | bir alarm |
| `outcomes` | Doktorun taburcuda işaretlediği gerçek sonuç (sepsis evet/hayır, başlangıç saati) | bir hasta |
| `audit_logs` | Kim, ne zaman, ne yaptı | bir olay |

---

## 2. Anahtarlar

### Birincil anahtar — Primary Key (PK)
Bir satırı **tekil olarak** tanımlayan sütun. Boş olamaz, tekrar edemez.

```sql
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,   -- PK: her hastaya otomatik artan numara
    ...
```
- `AUTOINCREMENT`: yeni satırda numara kendiliğinden verilir (1, 2, 3…); silinen numara tekrar kullanılmaz.
- **Bileşik (composite) PK**: birden fazla sütunun birlikte tekil olması.
  `measurements` tablosunda `PRIMARY KEY (observation_id, variable)` → aynı kayıtta "HR" iki kez olamaz.

### Yabancı anahtar — Foreign Key (FK)
Bir tablodaki sütunun **başka bir tablonun PK'sına** işaret etmesi. İlişkiyi kurar ve **geçersiz bağlantıyı engeller**.

```sql
CREATE TABLE observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,  -- FK
    entered_by INTEGER REFERENCES users(id) ON DELETE SET NULL,             -- FK
    ...
```
- Var olmayan bir hastaya (`patient_id = 999`) kayıt eklenmeye çalışılırsa veritabanı **reddeder**.
- SQLite'ta FK kontrolü varsayılan olarak kapalıdır; projede her bağlantıda `PRAGMA foreign_keys = ON` ile açılır (`db.connect()`).

**Silme davranışları (referential actions):**

| Kural | Anlamı | Projede |
|-------|--------|---------|
| `ON DELETE CASCADE` | Ana kayıt silinince bağlı kayıtlar da silinir | Hasta silinince ölçümleri, tahminleri, alarmları silinir |
| `ON DELETE SET NULL` | Ana kayıt silinince bağlantı boşaltılır, kayıt kalır | Kullanıcı silinse de girdiği ölçüm kalır, "giren" boş olur |
| (yok) | Bağlı kayıt varken ana kayıt silinemez | — |

### Aday anahtar / UNIQUE
PK olmayan ama yine de tekil olması gereken sütunlar.
- `users.username UNIQUE` → iki kullanıcı aynı adı alamaz.
- `patients.patient_id UNIQUE` → hastane protokol numarası tekrar edemez.
- `observations UNIQUE (patient_id, hour)` → bir hastanın aynı saati iki kez girilemez.

**PK ile UNIQUE farkı:** Bir tabloda tek PK olur ve NULL olamaz; UNIQUE birden fazla olabilir.
`patients.id` (yapay/surrogate anahtar) PK'dır, `patient_id` (doğal/natural anahtar) UNIQUE'tir. Böylece protokol
numarası değişse bile ilişkiler bozulmaz.

---

## 3. Kısıtlar (constraints) — veri bütünlüğü

| Kısıt | Örnek | Ne sağlar |
|-------|-------|-----------|
| `NOT NULL` | `name TEXT NOT NULL` | Adsız hasta olamaz |
| `CHECK` | `age CHECK (age BETWEEN 0 AND 120)` | Yaş 150 girilemez |
| `CHECK` | `role CHECK (role IN ('admin','doctor','nurse'))` | Tanımsız rol olamaz |
| `CHECK` | `status CHECK (status IN ('active','discharged'))` | Hasta durumu yalnızca iki değer |
| `CHECK` | `score CHECK (score BETWEEN 0 AND 1)` | Olasılık 0–1 dışında olamaz |
| `DEFAULT` | `status DEFAULT 'active'` | Değer verilmezse varsayılan |

Bu kısıtlar **uygulama kodundan bağımsız** çalışır: arayüzde bir hata olsa bile veritabanı yanlış veriyi kabul etmez.
Buna **savunma derinliği** (defense in depth) denir: doğrulama hem tarayıcıda, hem sunucuda, hem veritabanında yapılır.

---

## 4. Normalizasyon

Verinin tekrar etmeyecek ve tutarsızlığa düşmeyecek şekilde tablolara bölünmesi.

**Projenin eski hali (normalize değil):**
```
hourly_data: id | patient_id | hour | vital_signs = '{"HR": 118, "Temp": 38.7, ...}' | prediction | ...
```
Sorunlar: 38 değer tek bir metin alanında. SQL ile "laktatı 4'ün üstünde olan hastalar" sorgulanamaz, değer türü
kontrol edilemez, "bu değeri kim girdi" tutulamaz.

**Yeni hali:**
```
observations: id | patient_id | hour | entered_by | entered_at
measurements: observation_id | variable | value         ← her değer ayrı satır
predictions:  observation_id | score | risk_level | ...
```

| Normal form | Kural | Projede |
|-------------|-------|---------|
| **1NF** | Her hücrede tek (atomik) değer, tekrar eden grup yok | JSON metin → `measurements` tablosunda her değer bir satır |
| **2NF** | 1NF + PK'nın bir kısmına bağlı sütun yok | `measurements`'ın PK'sı (kayıt, değişken); `value` ikisine birden bağlı |
| **3NF** | 2NF + PK olmayan sütuna bağlı sütun yok | Hasta adı `observations`'ta tekrar edilmez, `patients`'ta bir kez durur |

`measurements` tablosu **EAV** (Entity–Attribute–Value) modelidir: değişken sayısı fazla ve çoğu boş olduğunda
(yoğun bakımda labların %90'ı ölçülmez) 38 sütunlu seyrek bir tablo yerine yalnızca ölçülen değerler saklanır.

**Bilinçli istisna:** `predictions.assessment` güvenilirlik ayrıntılarını JSON tutar. Bunlar sorgulanmaz,
yalnızca ekranda gösterilir. Sorgulanan alanlar (skor, risk, güvenilirlik) ayrı sütunlardadır.

---

## 5. İndeks

Kitap sonundaki dizin gibi: aramayı tüm tabloyu taramadan hızlandırır.

```sql
CREATE INDEX idx_obs_patient_hour ON observations(patient_id, hour);
CREATE INDEX idx_audit_time       ON audit_logs(timestamp);
```
- PK ve UNIQUE sütunlar için indeks otomatik oluşur.
- **Bedeli:** her ekleme/güncellemede indeks de güncellenir, disk alanı kaplar. Bu yüzden yalnızca sık aranan sütunlara konur.

---

## 6. JOIN — tabloları birleştirme

"Hastanın her saati, değerleri ve kaydı giren kişi":
```sql
SELECT o.hour, m.variable, m.value, u.full_name AS giren
FROM observations o
JOIN measurements m ON m.observation_id = o.id      -- INNER JOIN: iki tarafta da eşleşen
LEFT JOIN users u   ON u.id = o.entered_by          -- LEFT JOIN: kullanıcı yoksa da satır gelir
WHERE o.patient_id = ?
ORDER BY o.hour;
```
- **INNER JOIN**: yalnızca iki tabloda da karşılığı olanlar.
- **LEFT JOIN**: soldaki tablonun tüm satırları, sağda karşılık yoksa NULL.

---

## 7. View (görünüm)

Kayıtlı bir sorgu; tablo gibi kullanılır ama veri tutmaz.
```sql
CREATE VIEW v_patient_latest AS
SELECT p.id AS patient_pk, o.hour, pr.score, pr.risk_level, pr.reliability
FROM patients p
JOIN observations o ON o.id = (SELECT id FROM observations WHERE patient_id = p.id ORDER BY hour DESC LIMIT 1)
LEFT JOIN predictions pr ON pr.observation_id = o.id;
```
Hasta listesi her hastanın **son** riskini bu görünümden alır. Karmaşık sorgu bir kez yazılır, her yerde kullanılır.

---

## 8. Trigger (tetikleyici)

Bir olay olduğunda veritabanının kendiliğinden çalıştırdığı kod.
```sql
CREATE TRIGGER audit_logs_no_delete BEFORE DELETE ON audit_logs
BEGIN SELECT RAISE(ABORT, 'Denetim kayıtları silinemez'); END;
```
Denetim kayıtları **veritabanı seviyesinde** değiştirilemez ve silinemez. Uygulama kodunda bir açık olsa bile kayıtlar korunur.

---

## 9. Transaction ve ACID

**Transaction**: birlikte ya tamamen yapılan ya da hiç yapılmayan işlemler grubu.
Saatlik veri kaydında `observations` + `measurements` + `predictions` + `alarms` + `audit_logs` birlikte yazılır ve
tek bir `commit()` ile kalıcı olur. Ortada hata olursa hiçbiri yazılmaz; yarım kayıt oluşmaz.

| ACID | Anlamı | Örnek |
|------|--------|-------|
| **Atomicity** | Hepsi ya da hiçbiri | Ölçüm yazıldı ama tahmin yazılamadıysa ikisi de geri alınır |
| **Consistency** | Kısıtlar her zaman sağlanır | CHECK ve FK ihlali olan işlem reddedilir |
| **Isolation** | Eşzamanlı işlemler birbirini bozmaz | İki hemşire aynı anda farklı hastaya veri girer |
| **Durability** | Commit edilen veri kalıcıdır | Elektrik kesilse de kaydedilen ölçüm kaybolmaz |

**Taşıma (migration)**: eski veritabanı yeni şemaya taşınırken önce dosya yedeklenir, taşıma tek transaction'da yapılır,
hata olursa yedek geri yüklenir. Şema sürümü `PRAGMA user_version` ile tutulur.

---

## 10. Güvenlik

| Konu | Projede |
|------|---------|
| **SQL injection** | Kullanıcı girdisi sorguya asla metin olarak eklenmez; `?` parametreleri kullanılır: `execute('SELECT * FROM users WHERE username = ?', (username,))` |
| **Şifre saklama** | Düz metin değil, **tuzlu scrypt hash**. Aynı şifrenin iki hash'i farklıdır, hash'ten şifre geri çıkarılamaz |
| **Oturum** | Rastgele token HttpOnly çerezde; veritabanında yalnızca SHA-256 özeti. Veritabanı çalınsa bile oturumlar ele geçirilemez |
| **Yetki** | Rol bazlı (RBAC); her API isteğinde kontrol edilir |
| **Denetim izi** | Her işlem, kim/ne zaman/IP/hasta bilgisiyle; değiştirilemez |
| **Dosya erişimi** | `patients.db` tarayıcıdan indirilemez (sunucu yalnızca arayüz dosyalarını verir) |

---

## 11. Olası sorular ve kısa cevaplar

**Neden SQLite?**
Kurulum gerektirmez, tek dosyadır, Python'la gelir, ACID uyumludur. Tek bilgisayarda çalışan bir prototip için idealdir.
Çok sayıda eşzamanlı yazıcı olduğunda (hastane geneli) PostgreSQL'e geçilir; şema buna uygun tasarlandı.

**PK ile FK farkı nedir?**
PK bir satırı kendi tablosunda tekil tanımlar. FK başka bir tablonun PK'sına işaret ederek ilişki kurar ve
var olmayan kayda bağlantıyı engeller.

**Hasta silinince ne olur?**
`ON DELETE CASCADE` sayesinde ölçümleri, tahminleri ve alarmları silinir. Denetim kaydında FK yoktur, bu yüzden silme
olayı ve kimin sildiği kalır. Doktor kalıcı silemez, yalnızca taburcu eder (arşiv); kalıcı silme yöneticiye özeldir.

**Neden ölçümleri ayrı tabloya böldün?**
1NF: tek hücrede tek değer. Böylece değerler SQL ile sorgulanabilir, türü kontrol edilir, kimin girdiği tutulur.
Labların çoğu ölçülmediği için EAV yapısı boş sütunlarla dolu geniş bir tablodan daha verimlidir.

**Bire-bir ilişki var mı?**
İki tane:
- `observations` – `predictions`: her saatlik kaydın tek tahmini olur (`predictions.observation_id UNIQUE`).
- `patients` – `outcomes`: `outcomes.patient_id` hem **PK hem FK**. PK olduğu için tekrar edemez, FK olduğu için
  var olan bir hastaya işaret etmek zorunda; böylece her hastanın en fazla bir sonucu olur.

**Satırlar arası CHECK örneği?**
`outcomes` tablosunda `CHECK (sepsis = 'yes' OR onset_hour IS NULL)`: sepsis "Hayır" ise başlangıç saati girilemez.
Kısıt tek sütuna değil, sütunlar arası mantığa da konabilir.

**Veritabanından model için veri seti nasıl çıkıyor?**
Ölçümler `measurements` tablosunda her değer ayrı satır (uzun format). Dışa aktarmada her hasta-saat tek satıra
çevrilir (geniş format, PhysioNet ile aynı sütunlar), etiket `outcomes`'tan hesaplanır, ad ve protokol no yazılmaz.
Bu dönüşüme SQL/veri biliminde **pivot** denir.

**Çoka-çok ilişki nasıl kurulur?**
Ara tabloyla. Örneğin bir hasta birden çok doktora, bir doktor birden çok hastaya atanacaksa
`patient_doctors (patient_id FK, user_id FK, PK(patient_id, user_id))` eklenir.

**Veri tutarlılığını nasıl sağladın?**
Üç katmanda: tarayıcıda form doğrulama, sunucuda aralık ve yetki kontrolü, veritabanında NOT NULL, CHECK, UNIQUE, FK
ve tetikleyiciler.

**Aynı anda iki kişi aynı saati girerse?**
`UNIQUE(patient_id, hour)` ikinci kaydı ayrı satır olarak kabul etmez; uygulama aynı saati günceller ve eski değerleri
denetim kaydına yazar. SQLite yazma kilidiyle işlemleri sıraya koyar.

**Tablolarda kaç kayıt var, nereden bakarım?**
Yönetici olarak giriş → **Veritabanı** sayfası. Ya da DB Browser for SQLite ile `patients.db` açılır.

---

## 12. Gelecek: SQLite → PostgreSQL / Supabase

| | SQLite (şu an) | PostgreSQL (hastane sunucusu) | Supabase (bulut PostgreSQL) |
|---|---|---|---|
| Kurulum | Yok | Sunucu gerekir | Hesap açmak yeter |
| Eşzamanlı yazma | Tek yazıcı | Çok yazıcı | Çok yazıcı |
| Veri nerede | Bilgisayarda | Hastane içinde | Bulut (yurt dışı veri merkezi olabilir) |
| Hasta verisi için | Prototip | **Uygun** | KVKK açısından dikkat: yurt dışına aktarım, açık rıza, sözleşme |

Şema standart SQL ile yazıldığı için PostgreSQL'e geçiş büyük ölçüde tip adlarının değişmesinden ibarettir
(`INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`, `TEXT` tarihler → `TIMESTAMP`).
