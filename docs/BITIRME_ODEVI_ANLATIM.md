# SepsisAI — Bitirme Ödevi Proje Anlatımı

Bu belge projeyi baştan sona anlatır: **ne yapıyor, hangi dosyada ne var, hangi teknik nerede ve neden kullanıldı,
sonuçlar ne, sınırları ne.** Sunumda ve sorularda başvuru kaynağı olarak hazırlandı. Diğer belgeler daha dar
konulara odaklanır: model mimarisi için `MIMARI_ACIKLAMA.md`, veritabanı için `VERITABANI_REHBERI.md`, sunum akışı
için `SUNUM_REHBERI.md`.

---

## 1. Proje tek paragrafta

Yoğun bakımdaki hastanın saatlik vital bulgu ve laboratuvar değerlerinden, **sepsisin klinik olarak ortaya
çıkmasından 6 saat önce** uyarı vermeyi amaçlayan bir **karar destek sistemi**. Hemşire değerleri girer, sistem her
girişte riski (0–100), bu riskin **ne kadar güvenilir** olduğunu ve **neden** bu skoru verdiğini gösterir. Risk yüksekse
alarm açılır, doktor alarmı notla onaylar. Hasta taburcu edilirken doktor gerçek sonucu (sepsis oldu/olmadı) işaretler;
bu bilgi ileride modeli yeniden eğitmek için anonim veri setine dönüşür. Model, **sinir ağı (MLP) ve LightGBM'i
birleştiren hibrit** bir yapıdır: test ROC-AUC **0.847**, 5 katlı çapraz doğrulama **0.842 ± 0.009**.

**Önemli:** Bu bir araştırma prototipidir, klinik kullanım için onaylı değildir. Karar doktora aittir.

---

## 2. Veri

| | |
|---|---|
| Kaynak | **PhysioNet / Computing in Cardiology Challenge 2019** — “Early Prediction of Sepsis from Clinical Data” |
| Hastaneler | İki ABD hastanesi (Beth Israel Deaconess, Emory) |
| Boyut | **40.336 hasta, 1.552.210 saatlik satır** (`data/train.csv`) |
| Sütunlar | 8 vital bulgu, 26 laboratuvar, 6 demografik/idari bilgi, `SepsisLabel` |
| Pozitif oran | Satırların **%1.8**'i sepsis etiketli → çok dengesiz veri |
| Etiket | Sepsis-3 tanımına göre başlangıçtan **6 saat önce** 1 olur (erken tahmin için) |

**Eksik veri:** Laboratuvar değerlerinin çoğu saatlerin %90'ından fazlasında boş — çünkü her saat kan alınmaz. Bu
yüzden “değerin kendisi” kadar “ölçülüp ölçülmediği” ve “en son ne zaman ölçüldüğü” de bilgi taşır (aşağıda özellikler).

### Veri bölme — projenin en önemli dersi

Veri **hasta bazlı** bölündü: hastaların %60'ı eğitim, %20'si validasyon, %20'si test (seed 42, sepsis oranı her
parçada eşit → *stratified*). Bir hastanın hiçbir saati iki parçada birden bulunmaz (`scripts/train_v3.py →
split_patients`).

Neden önemli: Projenin ilk sürümü (v23) **0.88** AUC gösteriyordu. İnceleyince veri saat saat **rastgele**
bölündüğü ortaya çıktı: aynı hastanın 5. saati eğitimde, 6. saati testte. Üstelik **yaş + cinsiyet + hastaneye
yatıştan geçen süre** üçlüsü hastaların **%93**'ünü tek başına tanımlıyordu — model sepsisi değil hastayı
ezberliyordu. Hasta bazlı bölmede aynı model **0.80**'e düştü; gerçek değer buydu. Bu yüzden `HospAdmTime`
modelden tamamen çıkarıldı. Benzer şekilde v24'teki **0.9997**, etiketle ilişkili **sentetik biyobelirteçlerden**
(PCT, CRP) kaynaklanıyordu. Bu iki sızıntı bulunup düzeltildi; eski dosyalar `legacy/` altında belgelenerek duruyor.

**Kural:** Model seçimi, eşik seçimi, ağırlık seçimi, kalibrasyon — hepsi **validasyon** setinde yapıldı. Test seti
yalnızca en sonda **bir kez** kullanıldı.

---

## 3. Model gelişimi (hikâye)

| Sürüm | Yöntem | Bölme | ROC-AUC | Sonuç |
|---|---|---|---|---|
| v23 | GRU | rastgele pencere | 0.88 | sızıntı — geçersiz |
| v24 | GRU + sentetik biyobelirteç | — | 0.9997 | sızıntı — geçersiz |
| optimized v1 | medyan doldurma + sinir ağı | hasta bazlı | 0.809 | vital bulgulara tepkisiz |
| GRU v3 | ileri doldurma + ölçüm maskesi, 6 saat × 71 özellik | hasta bazlı | 0.806 | ilk dürüst model |
| **Hibrit v4** | **281 zaman serisi özelliği → LightGBM + MLP** | hasta bazlı | **0.847** | aktif model |

**Bulgu:** GRU, LSTM, farklı pencere uzunlukları gibi 7 mimari denendi; hepsi 0.80–0.81'de toplandı
(`scripts/experiments/`). Sorun mimaride değil, **modelin gördüğü bilgideydi**: 6 saatlik pencere uzun vadeli
eğilimleri göremiyordu ve GRU'lar 1–2 epokta hastaları ezberliyordu. Çözüm, eğilimleri açıkça hesaplayan
**özellik mühendisliği** oldu. Bu özelliklerle LightGBM tek başına 0.847'ye çıktı; aynı özelliklerle eğitilen sinir ağı
0.823 aldı; ikisini birleştirmek tek modelden daha **kararlı** sonuç verdi.

---

## 4. Özellik mühendisliği — `features_v4.py`

Her hasta saati için, **sadece o saate kadarki** verilerden 281 sayı hesaplanır (geleceğe bakmak yasak; testle
doğrulanıyor: `tests/test_features.py::test_no_future_leakage`).

| Grup | Adet | Ne | Neden |
|---|---|---|---|
| Son değer | 34 | Her klinik değişkenin en son ölçülen değeri (**ileri doldurma**) | Lab her saat ölçülmez; son bilinen değer geçerlidir |
| `_measured` | 34 | Bu saat ölçüldü mü (0/1) | Doktorun lab istemesi kendi başına bir kaygı işaretidir |
| `_since` | 34 | Son ölçümden geçen saat (hiç ölçülmediyse 72) | Değerin ne kadar “taze” olduğu |
| `_count` | 34 | Yatıştan beri kaç kez ölçüldü | İzlem yoğunluğu |
| Pencere istatistikleri | 84 | 6 vital (HR, SpO2, Ateş, SBP, MAP, Solunum) + 6 lab (Laktat, WBC, Kreatinin, Trombosit, BUN, FiO2) için **6 ve 24 saatlik** ortalama, min, maks, std | Kısa ve uzun vadeli eğilim |
| Değişim | 24 | 1 ve 6 saatlik fark (`_d1`, `_d6`) | Kötüleşme hızı |
| Yatıştan beri | 24 | tüm zamanların maks/min/ortalaması, şu anki değerin ortalamadan farkı | Hastanın kendi normaline göre sapma |
| Klinik türetilmiş | 9 | Şok indeksi (HR/SBP), nabız basıncı, BUN/Kreatinin, SpO2/FiO2, **SIRS**, kısmi **qSOFA**, kısmi **SOFA** ve 24 saatlik SOFA değişimi, laktat > 2 | Doktorların zaten kullandığı skorlar |
| Diğer | 4 | Yaş, cinsiyet, YBÜ saati, kayıt sayısı | |

Toplam: 34 × 4 + 84 + 24 + 24 + 9 + 4 = **281**.

**Ölçekleme (`V4Scaler`):** Eğitim setindeki medyan ve standart sapma ile z-skoru; eksikler medyanla doldurulur
(LightGBM eksik değeri kendisi işler, ölçekleme yalnızca sinir ağı için). Değerler ±6 standart sapmada kırpılır.

---

## 5. Hibrit model — `scripts/train_hybrid.py`

```
281 özellik ─┬─► LightGBM (3 seed, toplam 556 ağaç) ──► p₁ ─┐
             │                                              ├─► 0.70·p₁ + 0.30·p₂ ─► İzotonik kalibrasyon ─► P(sepsis)
             └─► Ölçekleme ─► Sinir ağı (MLP) ────────► p₂ ─┘
```

### Dal 1 — LightGBM (gradyan artırmalı karar ağaçları)
Her yeni ağaç, öncekilerin hatasını düzeltmeye çalışır. Parametreler: öğrenme hızı 0.03, 31 yaprak, yaprakta en az
1000 örnek (aşırı öğrenmeyi önler), her ağaçta özelliklerin %40'ı, satırların %80'i, L2 düzenlileştirme 20,
`scale_pos_weight=5` (sepsis örneklerine 5 kat ağırlık — dengesiz veri için). Erken durdurma validasyon **PR-AUC**'ye
göre (200 tur iyileşme yoksa dur). **3 farklı seed** (42, 7, 2024) ile eğitilip ortalaması alınır → varyans düşer.

### Dal 2 — Sinir ağı (çok katmanlı algılayıcı, MLP) — dersin yapay sinir ağı kısmı
| Katman | Görevi |
|---|---|
| `GaussianNoise(0.1)` | Girdiye küçük gürültü — ölçüm hatasına dayanıklılık, düzenlileştirme |
| `Dense(256, ReLU)` + L2(1e-4) | Tam bağlı katman, 281 → 256 |
| `BatchNormalization` | Katman çıktısını normalize eder, eğitimi hızlandırır |
| `Dropout(0.5)` | Her adımda nöronların yarısını rastgele kapatır — ezberlemeyi önler |
| `Dense(128, ReLU)` → BN → Dropout(0.5) | |
| `Dense(64, ReLU)` → Dropout(0.5) | |
| `Dense(1, Sigmoid)` | 0–1 arası olasılık |

Toplam **114.945** parametre. Eğitim: Adam (lr 0.001), **binary cross-entropy**, sınıf ağırlığı {0: 1, 1: 5},
batch 4096, en fazla 30 epok, **EarlyStopping** (validasyon PR-AUC, sabır 4, en iyi ağırlıklara dön),
**ReduceLROnPlateau** (iyileşme durursa öğrenme hızını yarıya indir). GPU: RTX 4060, birkaç dakika.

### Birleştirme ve kalibrasyon
- Ağırlık 0.70 / 0.30, **validasyon** PR-AUC'sini en yükseğe çıkaran değer olarak seçildi.
- **İzotonik regresyon** (`calibrator.pkl`): ham skoru gözlenen gerçek sepsis oranına çevirir. Ölçüm: beklenen
  kalibrasyon hatası (ECE) **0.048 → 0.002**. Yani ekranda “%30” yazıyorsa, bu tür hastaların gerçekten ~%30'u septik.

### Eşikler
| Eşik | Değer | Kullanım |
|---|---|---|
| Dengeli (F1) | 0.33 | Varsayılan “Yüksek risk” — duyarlılık %31, özgüllük %97 |
| Tarama | 0.066 | Validasyonda duyarlılık ≥ %80 olacak şekilde — test: duyarlılık %81, özgüllük %74 |

Risk seviyeleri (Çok düşük → Kritik) bu eşiklere göre belirlenir (`ml_service.get_risk_level`).

### Uygulamada LightGBM neden saf NumPy ile çalışıyor? — `gbm_numpy.py`
Windows'ta LightGBM'in C kütüphanesi ile TensorFlow aynı süreçte yüklenince **erişim ihlali (access violation)**
ile çöküyordu (iki kütüphanenin OpenMP sürümleri çakışıyor). Çözüm: eğitilmiş ağaçlar `trees.json`'a aktarılır ve
uygulamada **NumPy ile** dolaşılır. LightGBM'in kendi tahminiyle farkı **0.0** (testle doğrulanıyor:
`tests/test_gbm_numpy.py`). Yan fayda: uygulama LightGBM kurulmadan da çalışır. Eğitim betiklerinde ise `lightgbm`
her zaman sklearn/TensorFlow'dan **önce** import edilir.

---

## 6. Değerlendirme

### Metrikler — neden ROC-AUC yetmez?
- **ROC-AUC:** Rastgele bir septik ve bir sağlıklı saat seçilirse modelin septiğe yüksek skor verme olasılığı. 0.5 = yazı tura.
- **PR-AUC:** Pozitif oran %1.8 olduğu için daha zor ve dürüst metrik. Rastgele model 0.018 alır; bizimki **0.132** (≈7 kat).
- **Duyarlılık / özgüllük / PPV / NPV:** Seçilen eşikte kaç sepsis yakalanıyor, kaç yanlış alarm.
- **PhysioNet Utility:** Yarışmanın resmi metriği; zamanlamayı ölçer. Sepsisten 12–6 saat önceki alarmı ödüllendirir,
  geç alarmı ve yanlış alarmı cezalandırır. 1 = mükemmel, 0 = hiç alarm vermemek. `scripts/evaluate_clinical.py`
  içinde vektörel yazıldı ve resmi döngüsel tanımla birebir aynı olduğu testle doğrulandı (`tests/test_utility.py`).
- **ECE:** Kalibrasyon hatası.

### Sonuçlar (test, tek seferlik)
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Hibrit v4** | **0.847** | **0.132** |
| LightGBM dalı | 0.847 | 0.131 |
| MLP dalı | 0.823 | 0.113 |
| GRU v3 | 0.806 | 0.095 |

### 5 katlı çapraz doğrulama — `scripts/cv_hybrid.py`
Hastalar 5 gruba ayrıldı; her seferinde biri test, kalanlar eğitim (*GroupKFold*, hasta bazlı).
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **Hibrit** | **0.842 ± 0.009** | **0.129 ± 0.010** |
| LightGBM | 0.840 ± 0.010 | 0.128 ± 0.009 |
| MLP | 0.820 ± 0.006 | 0.112 ± 0.007 |

Düşük standart sapma: sonuç şanslı bir veri bölmesine bağlı değil.

### Klinikte kullanılan skorlarla karşılaştırma
| Yöntem | ROC-AUC | Utility |
|---|---|---|
| **Hibrit v4** | **0.847** | **0.427** |
| GRU v3 | 0.806 | 0.358 |
| NEWS2 (kısmi) | 0.682 | 0.115 |
| SIRS | 0.649 | 0.054 |
| qSOFA (kısmi) | 0.593 | 0.023 |

Model, hastanelerde bugün kullanılan skorlardan belirgin şekilde iyi. **Dikkat:** Yarışma kazananı 0.36 Utility aldı
ama **hiç görülmemiş üçüncü bir hastanede**; bizim 0.427 aynı iki hastanenin ayrılmış hastalarında. Doğrudan
karşılaştırılamaz — dürüstçe söylenmeli. qSOFA/NEWS2 bilinç (GKS) verisi olmadığı için kısmi.

### Hastaneler arası test — `scripts/hastane_arasi_test.py`
| Test edilen | Eğitim | ROC-AUC | PR-AUC | Utility | Tarama eşiği duyarlılık / özgüllük |
|---|---|---|---|---|---|
| Hastane A | Dış: yalnız hastane B | 0.787 | 0.103 | 0.359 | %76 / %66 |
| Hastane A | İç: kendi verisi | **0.817** | 0.121 | **0.421** | %77 / %70 |
| Hastane A | Birleşik: B + kendi | 0.816 | **0.126** | 0.417 | %80 / %67 |
| Hastane B | Dış: yalnız hastane A | 0.777 | 0.066 | 0.275 | **%59** / %81 |
| Hastane B | İç: kendi verisi | **0.849** | 0.115 | **0.378** | %77 / %76 |
| Hastane B | Birleşik: A + kendi | 0.846 | 0.114 | 0.359 | %72 / %80 |

PhysioNet verisi iki hastaneden gelir (A: 20.336 hasta, sepsis %8.8; B: 20.000 hasta, sepsis %5.7). Her hastanenin
hastalarının %30'u test için ayrıldı ve üç model aynı hastalarda karşılaştırıldı (`scripts/hastane_arasi_test.py`).
Eşikler her modelin kendi eğitim hastanesinde seçildi.

**Bulgular:**
- Hiç görülmemiş hastanede ROC-AUC **0.03–0.07 düşüyor** (B'de 0.849 → 0.777). Bu, eğitim veri miktarından değil
  hastane farkından: dış modeller iç modellerden daha çok hastayla eğitildi.
- **Eşik taşınmıyor:** A'da “%80 duyarlılık” diye seçilen eşik B'de yalnızca %59 yakalıyor.
- Hastanenin **kendi verisiyle** eğitince performans geri geliyor; diğer hastaneyi eklemek ek kazanç getirmiyor.
- Sonuç: yeni bir hastanede kullanmadan önce **yerel doğrulama ve yeniden eğitim şart**. Uygulamadaki taburcu sonucu →
  anonim veri seti döngüsü tam bunun için tasarlandı. Tek koşu olduğu için 0.01'lik farklar anlamlı değildir.

### Hasta düzeyinde test — `scripts/hasta_bazli_test.py`
Saat değil **hasta** sayılırsa (586 septik, 7.480 septik olmayan test hastası):
| Eşik | Etiket başlamadan alarm alan septik hasta | En az bir yanlış alarm alan hasta |
|---|---|---|
| Dengeli 0.33 | %33 | %3 |
| Tarama 0.066 | %80 | %58 |

### Klinik tutarlılık testi
Yapay “normal” ve “septik” vital bulgularla: 24. saatte normal hasta **%6.0**, septik hasta **%35.6**.
Tek parametre kötüleşince skor artıyor (ateş +14 puan, WBC +11, nabız +8.5). **Zayıf nokta:** MAP ve laktat
tek başına kötüleşince skor neredeyse değişmiyor — model bunları diğer bulgularla birlikte değerlendiriyor.

---

## 7. Güvenilirlik — `reliability.py`

Projenin başlangıç sorunu: **hiç değer girilmese bile model “hasta değil” diyordu.** Sebep: boş alanlar eğitim
medyanıyla dolduruluyordu, yani boş form = “ortalama sağlıklı hasta”. Çözüm katmanlı:

1. **Minimum veri kapısı:** HR, SpO2, ateş, solunum ve MAP'ten en az 3'ü yoksa risk verilmez → “Yetersiz Veri”.
2. **Güvenilirlik skoru (0–100)**, dört bileşenin ağırlıklı toplamı:
   - Veri tamlığı (%45): girilen değerlerin ağırlıklı oranı (vital %70, kritik lab %20, diğer %10)
   - Zaman kapsamı (%20): kaç saatlik geçmiş var
   - Model kararlılığı (%25): **MC-Dropout** — sinir ağı Dropout açıkken 30 kez çalıştırılır; tahminlerin
     tutarlılığına bakılır (Bayesçi belirsizlik yaklaşımı)
   - Dağılım uyumu (%10): eğitim verisinden |z| > 5 sapan değerler (dağılım dışı girdi)
3. **Klinik uyumsuzluk:** Model skoru eşiğin altındayken klinik kırmızı bayraklar (ör. düşük tansiyon, yüksek laktat,
   yüksek ateş) toplam 2 puanı geçiyorsa risk sınıfını klinik kurallar belirler, güvenilirlik en fazla 55 olur ve
   “model klinik tabloyla çelişiyor” uyarısı gösterilir (klinik geçersiz kılma).
4. **Girdi normalizasyonu:** FiO2 yüzde girilirse (40) kesre çevrilir (0.40); SBP+DBP varsa MAP hesaplanır.

---

## 8. Açıklanabilirlik — `explain.py`

Her tahminde **skoru en çok artıran 3 ve azaltan 2 faktör** gösterilir, ör. “Ateş 39.1 °C, 6 saatte +2.3 → +15 puan”.

Yöntem (**Saabas yolu katkıları**): Bir hasta her LightGBM ağacında kökten yaprağa bir yol izler. Her düğümde ağacın
o ana kadarki tahmin değeri bilinir; bir bölünmeden geçerken değer değişir ve bu değişim o bölünmedeki özelliğe yazılır.
Tüm ağaçlar toplanınca: `başlangıç değeri + Σ katkılar = modelin skoru` (testle doğrulanıyor). 281 özellik klinik
değişkene göre gruplanır (`Temp`, `Temp_d6`, `Temp_max24` → “Ateş”) ve logit katkısı yaklaşık puana çevrilir
(Δp ≈ p(1−p)·Δlogit). Ağaçlara düğüm değerlerini eklemek için LightGBM aynı parametrelerle yeniden eğitildi; sonuç
birebir aynı (`scripts/enrich_trees.py`). Açıklama, ağırlığı %70 olan LightGBM dalından gelir.

---

## 9. Web uygulaması

### Katmanlar ve dosyalar
| Dosya | Görevi |
|---|---|
| `app.py` | **Flask** — yalnızca URL'ler (route) ve JSON cevaplar |
| `security.py` | Giriş, oturum, roller, yetki dekoratörü `@require(...)` |
| `db.py` | **SQLite** — şema, göç (migration), tüm sorgular |
| `ml_service.py` | Modeli yükler, saatlik ızgarayı kurar, tahmin + güvenilirlik + açıklama döndürür |
| `features_v4.py`, `gbm_numpy.py`, `explain.py`, `reliability.py` | Model tarafı (yukarıda) |
| `config.py` | Model yolları, port, veritabanı yolu (ortam değişkeniyle değiştirilebilir) |
| `index.html`, `script.js`, `style.css` | Arayüz — saf HTML/CSS/JavaScript, framework yok |
| `login.html`, `login.js` | Giriş ekranı |
| `baslat.bat`, `sistem_kontrol.bat` (`sistem_kontrol.py`) | Tek tıkla başlatma ve 58 maddelik sistem kontrolü |

### Bir değer girildiğinde ne oluyor? (uçtan uca akış)
1. Hemşire formu doldurur → `script.js` `POST /api/patients/<id>/hourly-data` gönderir.
2. `security.require('observations.create')` oturumu ve rolü kontrol eder (değilse 401/403).
3. `ml_service.validate_vital_signs` değer aralıklarını kontrol eder (ör. HR 400 → 400 hatası, hangi alan yanlış).
4. `db.save_observation` → `observations` + `measurements` tablolarına yazılır; aynı saat tekrar girilirse eski
   değerler denetim kaydına konur.
5. `ml_service.assess` hastanın tüm saatlerini çeker, atlanan saatleri boş satırla doldurur (saatlik ızgara),
   `build_v4` ile 281 özelliği hesaplar, son saati modele verir.
6. LightGBM (NumPy) + MLP → ağırlıklı ortalama → kalibrasyon → risk seviyesi.
7. Güvenilirlik ve açıklama hesaplanır; sonuç `predictions` tablosuna yazılır.
8. Risk yüksekse `alarms` tablosuna alarm açılır. Her şey `audit_logs`'a işlenir.
9. Arayüz risk kartını, faktörleri, güvenilirliği ve risk trend grafiğini günceller.

### Roller ve yetkiler
| İşlem | Hemşire | Doktor | Yönetici |
|---|---|---|---|
| Hasta görme | ✓ | ✓ | ✓ |
| Hasta ekleme, değer girme | ✓ | ✓ | |
| Alarm onaylama (not zorunlu) | | ✓ | |
| Taburcu + sonuç etiketleme | | ✓ | |
| Hasta geçmişi | ✓ | ✓ | ✓ |
| Kullanıcı yönetimi, tüm denetim kayıtları, veri seti indirme, kalıcı silme | | | ✓ |

### Güvenlik
- Parolalar **scrypt** ile tuzlanmış özet olarak saklanır (düz metin asla).
- Oturum: rastgele jeton, tarayıcıda **HttpOnly + SameSite=Strict** çerez (JavaScript okuyamaz, başka siteden
  gönderilemez); veritabanında yalnızca jetonun **SHA-256** özeti.
- 30 dakika işlem yapılmazsa oturum düşer; 5 hatalı girişte hesap 15 dakika kilitlenir.
- Sunucu varsayılan olarak yalnızca `127.0.0.1`'de dinler.
- SQL sorguları her zaman parametreli (`?`) — SQL enjeksiyonuna kapalı.

### Veritabanı — SQLite, şema v3 (ayrıntı: `VERITABANI_REHBERI.md`)
| Tablo | İçerik | Önemli kısıtlar |
|---|---|---|
| `users` | kullanıcılar | `username` UNIQUE, `role` CHECK (nurse/doctor/admin) |
| `sessions` | açık oturumlar | FK → users |
| `patients` | hastalar | durum CHECK (active/discharged) |
| `observations` | hasta-saat kaydı | FK → patients, (hasta, saat) UNIQUE |
| `measurements` | her ölçülen değer ayrı satır | FK → observations — **normalizasyon**: 45 boş sütun yerine yalnızca girilen değerler |
| `predictions` | her tahmin | FK → observations |
| `alarms` | alarmlar, onaylayan doktor ve not | FK → patients, users |
| `outcomes` | taburcu sonucu (sepsis evet/hayır/belirsiz, başlangıç saati) | FK → patients |
| `audit_logs` | kim, ne zaman, ne yaptı | **UPDATE ve DELETE'i engelleyen trigger'lar** → değiştirilemez |
| `v_patient_latest` (view) | her hastanın son durumu | |

Yabancı anahtarlar açık (`PRAGMA foreign_keys=ON`). Eski şemadan geçişte önce yedek alınır, hata olursa geri yüklenir.

**Neden SQLite?** Tek dosya, kurulum yok, demo ve tek bilgisayar için ideal. Gerçek hastane (çok kullanıcı, ağ) için
PostgreSQL'e geçilmesi gerekir; şema standart SQL ile yazıldı; eski bir PostgreSQL taslağı `database/schema.sql`'de duruyor.

### Veri seti döngüsü
Taburcu sırasında doktor sonucu işaretler (sepsis evet/hayır/belirsiz ve başlangıç saati). Sepsis kesin ama başlangıç saati bilinmiyorsa “bilinmiyor” işaretlenir: sonuç kaydedilir ama hasta veri setine alınmaz, çünkü hangi saatlerin “sepsis öncesi” olduğu etiketlenemez. Yönetici **anonim CSV** indirir: isimler yok, hasta kimlikleri `L00001`
gibi, sütunlar PhysioNet formatında, `SepsisLabel` başlangıçtan 6 saat önce 1 (eğitim verisiyle aynı kural). Yani
hastane verisiyle modeli yeniden eğitmek mümkün (`scripts/veri_seti_disa_aktar.py`).

### Arayüz
Hasta listesi (aktif/arşiv), alarm paneli (açık/onaylanan), hasta detayı: risk kartı, **risk trend grafiği**, model
dalları (LightGBM/MLP ayrı skorlar), açıklama faktörleri, güvenilirlik, giriş formu. Form, öğretmen geri bildirimiyle
düzenlendi: model için **en önemli 11 değer** üstte (“Ana değerler”: 7 vital/oksijen + 4 kritik lab), diğerleri
“Ek değerler” bölümünde görünür ve doldurulabilir. “Model & Güvenilirlik” sayfası tüm metrikleri, çapraz doğrulamayı,
klinik karşılaştırmayı ve tutarlılık testini gösterir. Yönetici için kullanıcılar, denetim kayıtları, veritabanı sayfaları.

---

## 10. Testler — `tests/` (pytest, 64 test)

| Dosya | Neyi doğrular |
|---|---|
| `test_security.py` | parola özeti, kilitlenme, oturum zaman aşımı, rol yetkileri |
| `test_database.py` | FK/CHECK kısıtları, denetim kaydının silinememesi, göç |
| `test_features.py` | **geleceğe bakmama**, ileri doldurma, hastalar arası karışmama, 281 özellik |
| `test_gbm_numpy.py` | NumPy ağaçları = LightGBM; açıklama katkılarının toplamı = skor |
| `test_api.py` | boş form reddi, başlangıç saati bilinmeyen sepsisin veri setine girmemesi, aralık dışı değer, “Yetersiz Veri”, septik > normal, alarm notu zorunlu, taburcu sonucu, anonim veri seti |
| `test_utility.py` | Utility'nin resmi tanımla birebir aynı olması |

Ayrıca `sistem_kontrol.py`: uygulamayı uçtan uca deneyen 58 kontrol (demo öncesi çalıştırılır).

---

## 11. Kullanılan kütüphaneler ve teknolojiler

| Kütüphane | Nerede | Ne için |
|---|---|---|
| **TensorFlow / Keras 2.10** | `train_hybrid.py`, `ml_service.py` | Sinir ağı (MLP, önceki GRU), MC-Dropout |
| **LightGBM 4** | `train_hybrid.py`, `cv_hybrid.py` | Gradyan artırmalı ağaçlar (yalnızca eğitimde) |
| **NumPy** | her yerde, `gbm_numpy.py` | Diziler, uygulamada ağaç değerlendirme |
| **pandas** | `features_v4.py` | Hasta bazlı zaman serisi özellikleri (groupby, rolling, ffill) |
| **scikit-learn** | eğitim, `reliability.py` | Metrikler (ROC/PR-AUC), izotonik kalibrasyon, GroupKFold |
| **Flask** | `app.py` | Web sunucusu, REST API |
| **sqlite3** (standart) | `db.py` | Veritabanı |
| **hashlib / secrets** (standart) | `security.py` | scrypt parola özeti, güvenli jeton |
| **matplotlib** | eğitim betikleri | Eğitim eğrileri, değerlendirme grafikleri |
| **pytest** | `tests/` | Otomatik testler |
| HTML / CSS / JavaScript | arayüz | Framework'süz, `fetch` API, SVG trend grafiği |

Ortam: Python 3.10 (TensorFlow 2.10 Windows'ta 3.11+ desteklemez), GPU: NVIDIA RTX 4060.

---

## 12. Sınırlar (sorulursa dürüstçe)

1. **Başka hastanede performans düşüyor:** Hastaneler arası testte ROC-AUC 0.03–0.07 düştü ve eşik taşınmadı.
   Yeni hastanede yerel doğrulama ve yeniden eğitim gerekir. Türkiye'deki bir hastanede hiç test edilmedi.
2. **Duyarlılık–yanlış alarm dengesi:** Dengeli eşikte septiklerin üçte birini önceden yakalıyor; tarama eşiğinde
   %80'ini ama hastaların yarısından fazlasına en az bir yanlış alarm.
3. **Klinisyen davranışı öğreniliyor:** “Ölçüldü mü” ve “ne zamandır ölçülmüyor” özellikleri, doktorun kaygısını
   yansıtır. Bu bilgi değerli ama hastaneden hastaneye değişir.
4. **MAP ve laktatın tek başına etkisi zayıf.**
5. **GKS (bilinç) verisi yok** → qSOFA/NEWS2 kısmi.
6. Etiket (Sepsis-3) retrospektif tanımdır; gerçek başlangıç saati belirsiz olabilir.
7. Klinik kullanım için etik kurul, prospektif çalışma ve tıbbi cihaz onayı gerekir.

---

## 13. Gelecek çalışmalar
- Üçüncü bir kaynakla dış doğrulama (ör. MIMIC-IV, yerel hastane), alt grup analizi (yaş, cinsiyet).
- TreeSHAP ile kesin Shapley değerleri; conformal prediction ile garanti aralıklı belirsizlik.
- GRU-D / Transformer gibi eksik veriyi doğal işleyen zaman serisi modelleri.
- Toplanan anonim veriyle yeniden eğitim; PostgreSQL ve hastane ağında çalışma (HTTPS, HL7/FHIR entegrasyonu).

---

## 14. Olası sorular ve kısa cevaplar

**Neden ilk modelin 0.88'inden düşük bir sonuçla geldin?** 0.88 sızıntılıydı (aynı hasta eğitimde ve testte).
Hasta bazlı dürüst ölçümle ilk model 0.80; bizim 0.847 bunun üstüne gerçek bir iyileşme.

**Bu bir sinir ağı projesi mi, LightGBM daha iyi değil mi?** Sinir ağı modelin çekirdek dallarından biri ve
güvenilirlik ölçümü (MC-Dropout) sinir ağına dayanıyor. Tek başına LightGBM ile aynı ROC-AUC'yi veriyor ama
birleşim çapraz doğrulamada daha yüksek ve daha kararlı. Dürüst bulgu: bu tip tablo verisinde ağaç modelleri güçlü;
en iyisi ikisini birleştirmek.

**Dengesiz veriyle nasıl başa çıktın?** Sınıf ağırlığı (5 kat), PR-AUC'ye göre erken durdurma, eşiği validasyonda
seçme, PR-AUC ve Utility ile raporlama.

**Aşırı öğrenmeyi nasıl önledin?** Hasta bazlı bölme, Dropout, L2, BatchNorm, GaussianNoise, erken durdurma,
LightGBM'de yaprak başına en az 1000 örnek, 5 katlı CV ile doğrulama.

**Model neden bu skoru verdi?** Arayüzde her tahmin için en etkili faktörler puanlarıyla gösteriliyor (Saabas).

**Boş form girilirse ne olur?** Form reddedilir; 3'ten az temel vital varsa “Yetersiz Veri” — risk sınıfı verilmez.

**Veritabanı neden SQLite?** Demo ve tek makine için yeterli, normalize ve kısıtlı şema; hastane için PostgreSQL'e
geçiş planlı.

**Kayıtlar değiştirilebilir mi?** Denetim tablosunda UPDATE/DELETE trigger'la engelli; değer düzeltmelerinde eski
değer de kayda geçer.
