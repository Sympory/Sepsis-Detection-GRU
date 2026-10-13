# SepsisAI — Proje Özeti

**Proje:** GRU tabanlı yoğun bakım sepsis erken uyarı sistemi
**Ders:** Python Programlama
**Geliştirici:** Ahmet Sami Ceyhan

---

## 1. Amaç

Yoğun bakımdaki hastaların saatlik **vital bulgu ve laboratuvar** verilerinden sepsis riskini **6 saat önceden**
tahmin eden bir derin öğrenme modeli geliştirmek, bu modeli bir **veritabanı** ve **web arayüzü** ile kullanılabilir hale getirmek.

Sepsis, enfeksiyona karşı vücudun kontrolsüz yanıtıdır. Yoğun bakımda önde gelen ölüm nedenlerindendir ve
her saatlik gecikme riski artırır.

---

## 2. Veri

| | |
|---|---|
| Kaynak | PhysioNet / Computing in Cardiology Challenge 2019 (Reyna ve ark., 2020) |
| Boyut | **40.336 hasta, 1,55 milyon saatlik kayıt** |
| Değişkenler | 34 klinik değişken (nabız, ateş, tansiyon, SpO2, laktat, WBC…) + yaş, cinsiyet, saat |
| Zorluklar | Sepsis oranı yalnızca **%1.8**; laboratuvar değerlerinin **%90'dan fazlası eksik** |

---

## 3. Sistem mimarisi

```
┌──────────────────────────────┐
│  Web Arayüzü (HTML/CSS/JS)   │  ← kullanıcı veri girer, riski görür
└──────────────┬───────────────┘
               │ REST API (JSON)
┌──────────────▼───────────────┐
│  Flask Sunucusu (app.py)     │  ← istekleri karşılar, doğrular
└───┬──────────────┬───────────┘
    │              │
┌───▼──────┐  ┌────▼──────────────────────────────┐
│ SQLite   │  │ Yapay Zekâ Katmanı                │
│ Veri-    │  │  features.py    → veri hazırlama  │
│ tabanı   │  │  GRU modeli     → risk tahmini    │
│          │  │  reliability.py → güvenilirlik    │
└──────────┘  └───────────────────────────────────┘
```

---

## 4. Kullanılan Python kütüphaneleri

| Kütüphane | Projedeki görevi |
|-----------|------------------|
| **TensorFlow / Keras** | GRU sinir ağını kurma, eğitme ve tahmin yapma |
| **pandas** | 1,55 milyon satırlık veriyi okuma, hasta bazlı gruplama, eksik değerleri doldurma |
| **NumPy** | Matris işlemleri, 6 saatlik pencerelerin oluşturulması |
| **scikit-learn** | İzotonik kalibrasyon, ROC-AUC / PR-AUC metrikleri |
| **Flask** | Web sunucusu ve REST API |
| **flask-cors** | Tarayıcı–sunucu iletişim izinleri |
| **sqlite3** | Normalize veritabanı: kullanıcılar, hastalar, ölçümler, tahminler, alarmlar, denetim kaydı |
| **Werkzeug** | Şifrelerin güvenli hash'lenmesi (scrypt) |
| **hashlib / secrets** | Oturum token'ı üretimi ve özetlenmesi |
| **matplotlib** | Eğitim grafikleri, ROC ve kalibrasyon eğrileri |
| **LightGBM** | Karşılaştırma için ağaç tabanlı model |
| **pickle / json** | Model ön işleme nesnelerinin ve sonuçların kaydı |
| **Chart.js** *(JavaScript)* | Arayüzdeki risk trendi grafiği |

---

## 5. Kullanılan teknikler

### Veri hazırlama
| Teknik | Neden |
|--------|-------|
| Hasta içinde **ileri doldurma** | Lab değerleri her saat ölçülmez; son ölçülen değer geçerli kabul edilir |
| **Ölçüm maskeleri** (0/1) | Modele “bu değer bu saat ölçüldü mü” bilgisi ayrıca verilir |
| **Z-skoru standartlaştırma** | Farklı birimlerdeki değerleri aynı ölçeğe getirmek |
| **Zaman serisi özellikleri** (281) | Eğilimi modele açıkça vermek: 6/24 saatlik ve yatıştan bu yana ortalama-min-maks, değişim, son ölçümden geçen süre, SIRS/SOFA/şok indeksi |
| **Hasta bazlı bölme** (%60/%20/%20) | Aynı hasta hem eğitimde hem testte olmasın, yani **veri sızıntısı** olmasın |

### Model — Hibrit: yapay sinir ağı + LightGBM
```
281 zaman serisi özelliği
  ├─ Sinir ağı (MLP): Dense 256 → Dense 128 → Dense 64 → Sigmoid
  │    (BatchNorm, Dropout 0.5, Gaussian gürültü, L2) — 114.945 ağırlık
  └─ LightGBM: 3 seed × ~185 karar ağacı (uygulamada saf NumPy ile)
  → 0.3 × sinir ağı + 0.7 × LightGBM → izotonik kalibrasyon → Sepsis olasılığı
```
İlk model bir **GRU** (tekrarlayan sinir ağı) idi: son 6 saatin ham değerlerini görüyordu (0.806). Eğilimleri açıkça
hesaplayan özellikler ve iki modelin birleşimiyle **0.847**'ye çıkıldı. Ayrıntı: `docs/MIMARI_ACIKLAMA.md`.

### Eğitim
| Teknik | Neden |
|--------|-------|
| **Dropout, L2, Gaussian gürültü** | Aşırı öğrenmeyi (ezberlemeyi) önlemek |
| **Sınıf ağırlığı** | Sepsis örnekleri çok az (%1.8) |
| **Erken durdurma** | Doğrulama skoru düşünce eğitimi durdurmak |
| **Adam optimizer, gradyan kırpma** | Kararlı ve hızlı öğrenme |
| **GPU ile eğitim** | Eğitim ~3 dakika (RTX 4060) |

### Tahmin sonrası
| Teknik | Neden |
|--------|-------|
| **İzotonik kalibrasyon** | Model skorunu gerçek gözlenen orana çevirmek |
| **MC-Dropout** (30 tahmin) | Modelin ne kadar emin olduğunu ölçmek (belirsizlik aralığı) |
| **Güvenilirlik skoru** (0–100) | Veri tamlığı + zaman kapsamı + model kararlılığı + dağılım uyumu |
| **Minimum veri kapısı** | 3'ten az temel vital varsa tahmin vermemek (“Yetersiz Veri”) |
| **Klinik kurallar** | Düşük tansiyon, yüksek laktat gibi bulgularda riski yükseltmek |

---

## 6. Veritabanı, kullanıcılar ve güvenlik

### Normalize SQLite şeması (8 tablo)

| Tablo | İçerik | Önemli kısıtlar |
|-------|--------|-----------------|
| `users` | Kullanıcı hesapları | `username` UNIQUE, `role` CHECK (yönetici/doktor/hemşire) |
| `sessions` | Açık oturumlar | token SHA-256 özeti, `user_id` → users |
| `patients` | Hastalar | `patient_id` UNIQUE, yaş CHECK 0–120, durum CHECK (yatıyor/taburcu) |
| `observations` | Saatlik kayıtlar | UNIQUE(hasta, saat), kaydı giren kullanıcı |
| `measurements` | Ölçüm değerleri (her değer ayrı satır) | PK(kayıt, değişken) |
| `predictions` | Model tahmini, risk, güvenilirlik, model sürümü | kayıt başına bir tahmin |
| `alarms` | Yüksek risk uyarıları | onaylayan doktor, zaman, klinik not |
| `audit_logs` | Denetim kaydı | **tetikleyicilerle değiştirilemez/silinemez** |

- Yabancı anahtarlar, CASCADE silme, indeksler, bir görünüm (`v_patient_latest`) ve tetikleyiciler (trigger) kullanılıyor.
- Eski sürümün veritabanı ilk açılışta **otomatik yedeklenip** yeni şemaya taşınıyor.

### Roller

| İşlem | Hemşire | Doktor | Yönetici |
|-------|:------:|:------:|:-------:|
| Hasta kabul, saatlik veri girişi | ✓ | ✓ | — |
| Alarm onayı ve klinik not | — | ✓ | — |
| Taburcu / arşiv | — | ✓ | — |
| Kalıcı silme, kullanıcı yönetimi, tüm denetim kayıtları | — | — | ✓ |

### Güvenlik
- Şifreler tuzlu **scrypt** hash, oturum **HttpOnly** çerez, **30 dk** hareketsizlikte oturum kapanır.
- **5 hatalı** girişte hesap **15 dk** kilitlenir. Her API ucunda rol kontrolü yapılır, yetkisiz denemeler kaydedilir.
- Denetim kaydı: giriş/çıkış, başarısız giriş, veri girişi (üzerine yazmada eski ve yeni değerler), alarm, taburcu, silme, kullanıcı işlemleri.

---

## 7. Web arayüzü

| Sayfa | Özellikler |
|-------|-----------|
| **Genel Bakış** | Hasta sayıları, en yüksek riskten sıralı hasta listesi, alarm bandı |
| **Hastalar** | Arama, ekleme, silme |
| **Hasta Detay** | Risk trendi grafiği (belirsizlik bandıyla), saatlik veri giriş formu, risk ve güvenilirlik paneli, “Neden?” açıklamaları, kayıt geçmişi |
| **Alarmlar** | Yüksek riskli, yetersiz verili ve düşük güvenilirlikli hastalar |
| **Model & Güvenilirlik** | Ölçülmüş metrikler, eşik seçenekleri, model karşılaştırması, klinik tutarlılık testi |

Ek olarak:
- Form doğrulama: zorunlu alanlar, aralık uyarıları, birim dönüşümü
- Canlı veri tamlığı göstergesi
- Açık/koyu tema ve mobil uyumlu tasarım
- Tek tıkla başlatma: `baslat.bat`
- Otomatik sistem testi: `sistem_kontrol.bat`

---

## 8. Sonuçlar

Değerler, eğitimde **hiç görülmemiş 8.066 test hastası** üzerinde ölçüldü.

| Metrik | Değer |
|--------|-------|
| ROC-AUC | **0.847** (ilk GRU: 0.806) |
| PR-AUC | **0.132** (rastgele tahminin ~7 katı; ilk GRU: 0.095) |
| Kalibrasyon hatası | 0.048 → **0.002** |
| Normal / septik hasta skoru (24. saat) | **%6 / %36** |
| Tarama modunda özgüllük (%80 duyarlılıkta) | %64 → **%75** (yanlış alarm ~%30 az) |
| 5 katlı çapraz doğrulama | ROC-AUC **0.842 ± 0.009**, PR-AUC 0.129 ± 0.010 |
| PhysioNet Utility (iç test) | **0.427** (ilk GRU 0.358; NEWS2 0.115, SIRS 0.054, qSOFA 0.023) |
| Klinik skorlarla AUC | Hibrit 0.847 · NEWS2 0.682 · SIRS 0.649 · qSOFA 0.593 |

### Geliştirme sürecinde öğrenilenler
| Sürüm | Sonuç | Ders |
|-------|-------|------|
| İlk model | 0.880 | Veri rastgele bölünmüştü, aynı hasta testte de vardı → **iyimser sonuç** |
| 56 özellikli model | 0.9997 | Veri setindeki bazı biyobelirteçler sentetikti, etiketi ele veriyordu → **sızıntı** |
| Eski model | 0.809 | Septik hastaya %7 veriyordu: “değer ne?” yerine “ölçüldü mü?” öğrenmişti |
| GRU v3 | 0.806 | Klinik olarak tutarlı; eksik veride tahmin vermiyor |
| **Hibrit v4 (güncel)** | **0.847** | Eğilimler açıkça hesaplanınca hem sinir ağı hem ağaç modeli belirgin iyileşti |

**Ana ders:** Yüksek skor her zaman iyi model demek değildir. Doğru değerlendirme yöntemi en az model kadar önemlidir.

---

## 9. İleride yapılacaklar (6 haftalık plan)

| Hafta | Konu | Kullanılacak teknikler / kütüphaneler |
|-------|------|---------------------------------------|
| **1** | Değerlendirme altyapısı | 5 katlı çapraz doğrulama (GroupKFold), PhysioNet Utility skoru, qSOFA/NEWS2 ile karşılaştırma |
| **2** | Özellik mühendisliği | Son ölçümden geçen süre, 12/24 saatlik trendler, SOFA skoru, şok indeksi |
| **3** | Yeni mimariler | Daha uzun pencere (12–48 saat), GRU-D, TCN, Transformer; **Optuna** ile hiperparametre arama |
| **4** | Model birleştirme | GRU + LightGBM topluluğu (ensemble), focal loss, kalibrasyon karşılaştırması |
| **5** | Açıklanabilirlik | **SHAP** ile “bu skoru en çok artıran faktörler”, conformal prediction, deep ensembles |
| **6** | Doğrulama ve teslim | Farklı hastanede test (A → B), alt grup analizi, rapor |

### Veritabanı ve uygulama geliştirmeleri

Giriş ekranı, roller, normalize veritabanı ve denetim kaydı **tamamlandı** (Bölüm 6). Kalanlar:

| Geliştirme | Açıklama |
|------------|----------|
| **PostgreSQL** | Çok kullanıcılı eşzamanlı yazma için; **SQLAlchemy** ORM ile geçiş (şema `database/schema.sql`'de hazır) |
| **Hastane entegrasyonu** | HBYS ve laboratuvardan HL7/FHIR ile otomatik veri akışı |
| **Test ve dağıtım** | **pytest** ile birim testleri, **Docker** ile paketleme, HTTPS |
| **İki aşamalı doğrulama** | Hastane ağı dışında erişim gerekirse |

**6 hafta sonundaki hedefler:**
- ROC-AUC 0.83–0.85 (literatürde benzer çalışmalar yaklaşık 0.80–0.85)
- Utility ≥ 0.30 (yarışma kazananı 0.36)
- Tahmin başına açıklama
- Çok kullanıcılı, denetlenebilir veritabanı

---

## 10. Klinik uygulamaya geçiş yol haritası

Proje bir hekimle birlikte yürütülmektedir. Sistemin hastanede kullanılabilmesi için yol, model geliştirmekten çok
**yerel veri, klinik doğrulama ve mevzuat** üzerinden geçer. Bu süreç genellikle 2–4 yıl sürer.

### 10.1 Etik ve hukuki zemin
- **Etik kurul onayı:** Hastanenin geriye dönük verisi için Girişimsel Olmayan Klinik Araştırmalar Etik Kurulu.
  İleriye dönük çalışmalar için ayrıca onay alınır.
- **KVKK:** Sağlık verisi “özel nitelikli kişisel veri”dir. Anonimleştirme, veri işleme sözleşmesi ve verinin
  hastane dışına çıkmaması gerekir.

### 10.2 Yerel veri ve etiketleme
- PhysioNet verisi ABD hastanelerinden gelir. Türkiye'deki yoğun bakımda hasta profili, ölçüm sıklığı ve birimler
  farklı olabilir (ör. kreatinin mg/dL veya µmol/L).
- Hastane bilgi yönetim sisteminden (HBYS) geriye dönük, binlerce hastalık bir veri seti çıkarılır.
- Sepsis başlangıcı **Sepsis-3** kriterlerine (SOFA ≥ 2 artış + şüpheli enfeksiyon) göre belirlenir. Belirsiz vakalara
  ikinci bir hekim bakar. Etiket kalitesi, model kalitesini algoritmadan daha çok belirler.

### 10.3 Dış doğrulama ve yerel uyarlama
- Mevcut model önce **hiç değiştirilmeden** yerel veride test edilir. Bu, gerçek genellenebilirliği gösterir.
- Ardından yerel veriyle yeniden kalibre edilir veya yeniden eğitilir.
- Uyarıcı örnek: Yaygın kullanılan Epic Sepsis Model, bağımsız dış doğrulamada beklenenden çok daha düşük performans
  gösterdi (AUC ≈ 0.63; Wong ve ark., *JAMA Internal Medicine*, 2021).
- Sonuçlar **TRIPOD+AI** kılavuzuna göre raporlanır.

### 10.4 Sessiz mod (prospektif, hastaya etkisiz)
- Sistem gerçek zamanlı çalışır ama **uyarılar hekimlere gösterilmez**, yalnızca kaydedilir.
- Birkaç ay boyunca şunlar ölçülür: model sepsisten kaç saat önce uyardı, hasta başına günde kaç alarm verdi,
  gerçek klinik akışta veri eksikliği ne kadar.
- **Alarm yorgunluğu** bu aşamada değerlendirilir. Çok alarm veren sistem kısa sürede görmezden gelinir.

### 10.5 Klinik pilot ve etki çalışması
- Uyarılar hekimlere gösterilir, hekim geri bildirimi toplanır (“bu alarm yerinde miydi?”).
- Erken klinik değerlendirme **DECIDE-AI** kılavuzuna göre raporlanır.
- Asıl soru “AUC kaç?” değil, **“hasta sonuçları iyileşti mi?”** sorusudur: antibiyotiğe başlama süresi,
  mortalite, yoğun bakım süresi.
- Olumlu örnek: Johns Hopkins'in TREWS sistemi çok merkezli prospektif çalışmada, uyarıya hızlı yanıt verilen
  hastalarda mortalitenin azaldığını gösterdi (Adams ve ark., *Nature Medicine*, 2022).
- En güçlü kanıt randomize veya kademeli geçişli (stepped-wedge) bir çalışmadır.

### 10.6 Mevzuat: tıbbi cihaz yazılımı
- Tanı/tedavi kararına destek veren yazılım **tıbbi cihaz yazılımı (SaMD)** sayılır.
- Türkiye'de TİTCK Tıbbi Cihaz Yönetmeliği, AB MDR 2017/745 ile uyumludur. Bu tür bir karar destek yazılımı büyük
  olasılıkla **Sınıf IIa veya IIb** kapsamındadır ve onaylanmış kuruluş denetimiyle CE işareti gerektirir.

| Standart | Konu |
|----------|------|
| ISO 13485 | Kalite yönetim sistemi |
| IEC 62304 | Tıbbi yazılım yaşam döngüsü |
| ISO 14971 | Risk yönetimi |
| IEC 62366 | Kullanılabilirlik mühendisliği |

- Ara yol: Araştırma aşamasında sistem, etik kurul onayıyla ve “araştırma amaçlı, klinik karar için kullanılmaz”
  etiketiyle çalıştırılabilir.

### 10.7 Teknik altyapı (hastane entegrasyonu)

| Konu | Gerekli olan |
|------|--------------|
| Veri akışı | HBYS ve laboratuvardan **otomatik** veri çekme (HL7 / FHIR). Elle veri girişi gerçekçi değildir. |
| Güvenlik | Kullanıcı girişi, rol bazlı yetki, denetim kaydı, şifreleme, hastane içi sunucu |
| Süreklilik | Kesintisiz çalışma, yedekleme, hata durumunda güvenli davranış |
| İzleme | Veri kayması takibi, aylık performans raporu, gerektiğinde yeniden kalibrasyon |
| Sürüm yönetimi | Her tahminde kullanılan model sürümünün kaydı |
| Açıklanabilirlik | “Bu alarm neden çıktı?” sorusuna yanıt (SHAP vb.) |

### 10.8 Hekimin rolü
- **Klinik soruyu tanımlamak:** Hangi hastada, ne zaman, hangi eyleme yol açacak bir uyarı isteniyor?
- **Eylem protokolü:** Uyarı geldiğinde ne yapılacak (laktat, kültür, antibiyotik değerlendirmesi)?
  Eylemi olmayan alarm işe yaramaz.
- **Etiketleri doğrulamak**, etik kurul başvurusunu yürütmek, hekim geri bildirimini toplamak.
- **Kabul edilebilir yanlış alarm oranını** ve buna göre karar eşiğini belirlemek.

### 10.9 Zaman çizelgesi

| Faz | Süre | Çıktı |
|-----|------|-------|
| 0. Mevcut prototip | Tamamlandı | PhysioNet verisiyle çalışan sistem |
| 1. Etik kurul + yerel veri | 3–6 ay | Etiketlenmiş yerel veri seti |
| 2. Dış doğrulama + yerel uyarlama | 2–3 ay | TRIPOD+AI uyumlu yayın |
| 3. Sessiz mod | 3–6 ay | Gerçek zamanlı performans ve alarm yükü |
| 4. Klinik pilot | 6–12 ay | Hasta sonuçlarına etkisi |
| 5. Mevzuat (CE / TİTCK) | 1–2 yıl (paralel) | Tıbbi cihaz onayı |

---

## 11. Çalıştırma

```
baslat.bat              → uygulamayı başlatır, tarayıcıyı açar
sistem_kontrol.bat      → uçtan uca test
py -3.10 scripts/train_v3.py           → modeli yeniden eğitir (~3 dk)
py -3.10 scripts/demo_hastalari.py     → örnek hastalar ekler
```

> ⚠️ Araştırma ve eğitim amaçlıdır; klinik kullanım için onaylı değildir.
