# Sepsis Erken Tespit Projesi — Değerlendirme ve Yapılacaklar

*Tarih: 8 Ekim 2026 · İncelenen: `app.py`, `config.py`, ön yüz, `prepare_sequence_dataset_v23.py`, `models/`, `data/processed/`*

Bu belgedeki sayılar aktif model (`models/optimized/best_model.keras`) üzerinde, mevcut
`data/processed/` test setinde (270.657 pencere, %1.74 pozitif) yeniden ölçülmüştür.
Ölçümleri tekrar üretmek için: `python scripts/fit_calibration.py`.

> **Güncelleme (aynı gün): Yeni model v3 eğitildi ve aktif model yapıldı.** Aşağıdaki 1–5. bölümler eski
> modelin (Optimized v1) incelemesidir. v3 sonuçları ve 0.88'in kaynağı için **Bölüm 7**'ye bakın.

---

## 1. Özet

| Alan | Durum | Not |
|---|---|---|
| Veri hazırlama | 🟢 İyi | Hasta bazlı bölme (GroupShuffleSplit) ile veri sızıntısı önlenmiş |
| Model performansı | 🔴 Zayıf | ROC-AUC 0.81, PR-AUC 0.089, 0.24 eşiğinde **duyarlılık %31** |
| Modelin öğrendiği sinyal | 🔴 Sorunlu | Vital bulgulara neredeyse tepkisiz; YBÜ saatine ve “lab ölçülmüş mü” bilgisine dayanıyor |
| Tahmin güvenilirliği | 🟢 Eklendi | Minimum veri kapısı, MC-Dropout, kalibrasyon, tamlık skoru |
| Web arayüzü | 🟢 Yenilendi | Risk + güvenilirlik paneli, canlı veri tamlığı, açık/koyu tema |
| Güvenlik | 🔴 Eksik | Demo giriş her şifreyi kabul ediyor; statik dosya sızıntısı kapatıldı |
| Dokümantasyon tutarlılığı | 🟡 Düzeltildi | UI/config'de olmayan attention katmanı ve 0.88 AUC iddiası vardı |

---

## 2. “Hasta girmediğim halde hasta değil diyordu” — kök neden

1. **Eksik alanlar medyanla dolduruluyor.** `SimpleImputer(strategy='median')` boş bırakılan
   her alanı eğitim setinin medyanıyla doldurur: HR 83.5, Ateş 37.0, MAP 80, Laktat 1.8, WBC 10.3…
   Yani boş form, model için “tamamen ortalama bir YBÜ hastası” demektir.
2. **Model bu “ortalama hasta”ya %5.4 skor veriyor**, bu da “Çok Düşük (Stabil)” etiketine düşüyordu.
3. Arayüz ve backend boş gönderimi engellemiyordu, sonuçla birlikte bir güven ölçüsü de göstermiyordu.

### Yapılan düzeltme

| Katman | Ne yapıldı |
|---|---|
| Backend | Hiç ölçüm yoksa istek reddediliyor (400). HR, SpO2, Ateş, Solunum, MAP'ten 3'ünden azı girildiyse risk sınıfı verilmiyor: **“Yetersiz Veri”** |
| Backend | Her tahmine 0–100 **güvenilirlik skoru** ve gerekçe listesi ekleniyor (`reliability.py`) |
| Ön yüz | Canlı veri tamlığı çubuğu; yetersiz veride “Sonuç yok: hastanın sepsis olmadığı anlamına gelmez” mesajı |

---

## 3. Güvenilirlik nasıl hesaplanıyor (`reliability.py`)

| Bileşen | Ağırlık | Açıklama |
|---|---|---|
| Veri tamlığı | %45 | Gerçekten girilen değerlerin ağırlıklı oranı: temel vitaller %70, kritik lab (Laktat, WBC, Kreatinin, Trombosit, Bilirubin, pH) %20, diğerleri %10 |
| Zaman kapsamı | %20 | Model 6 saatlik pencere bekler; eksik saatler ilk ölçümün kopyasıdır |
| Model kararlılığı | %25 | **MC-Dropout**: Dropout açık, 30 stokastik ileri geçiş. Kaçının eşiğin aynı tarafında kaldığı ölçülür. Ayrıca %90 belirsizlik aralığı gösterilir |
| Dağılım uyumu | %10 | Eğitim dağılımından \|z\| > 5 uzaklıktaki değerler |

Ek kurallar:
- **Minimum veri kapısı:** < 3 temel vital → skor en fazla 25, seviye “Yetersiz Veri”.
- **Model–klinik çelişkisi:** Klinik kırmızı bayraklar ≥ 2 puan iken model skoru eşiğin altındaysa
  risk sınıfını klinik kurallar belirler, model güvenilirliği en fazla “Orta” olur ve bu durum açıkça yazılır.
- **Kalibrasyon:** Validasyon setinde izotonik regresyon (`models/optimized/calibrator.pkl`).
  Test ECE: **0.084 → 0.002**, Brier: 0.0246 → 0.0165.

Doğrulama (test setinden 150 pozitif + 150 negatif örnek):
model kararlı olduğunda doğruluk **%62**, kararsız olduğunda **%37.5**. MC-Dropout belirsizliği gerçekten bilgi taşıyor.

---

## 4. Model değerlendirmesi

### 4.1 Ölçülmüş performans (test seti)

| Çalışma noktası | Eşik | Duyarlılık | Özgüllük | PPV | NPV |
|---|---|---|---|---|---|
| Mevcut (F1 odaklı) | 0.240 | **%31.3** | %95.7 | %11.5 | %98.7 |
| Tarama (val. duyarlılık ≥ %80) | 0.095 | %82.3 | %61.7 | %3.7 | %99.5 |

- Mevcut eşikte sepsis pencerelerinin **%69'u kaçırılıyor**. “Düşük risk” sonucu tek başına güvence vermez.
- Erken uyarı sistemi için tarama eşiği daha uygun, ancak yanlış alarm yükü ciddi şekilde artar.
  Bu bir klinik karar olduğu için eşik değiştirilmedi. İki seçenek de *Model & Güvenilirlik* sayfasında gösteriliyor.

### 4.2 Diğer kayıtlı modeller (aynı 40 özellikli test alt kümesi)

| Model | ROC-AUC | PR-AUC | Not |
|---|---|---|---|
| optimized/best_model (aktif) | 0.797 | 0.089 | En iyi kalibre (Brier 0.024) |
| gru_v25_focal | 0.802 | 0.089 | Focal loss → olasılıklar şişkin |
| paper_model_v1 | 0.801 | 0.093 | |
| paper_model_final/final_best | 0.794 | 0.071 | |
| paper_model_final/final_best_balanced | 0.796 | 0.089 | Ortalama skor 0.61, kalibrasyonu çok bozuk |
| gru_v23_best | — | — | **41 özellik** bekliyor, mevcut 40 özellikli pipeline ile çalışmıyor |

Raporlarda geçen **0.88 ROC-AUC** değeri `gru_v23_best` modeline aittir. O sürüm veriyi saat saat
**rastgele** bölüyordu, yani aynı hasta hem eğitimde hem testte vardı (ayrıntı: Bölüm 7). Hasta bazlı
değerlendirmede tüm modeller ~0.80 bandındadır.

### 4.3 Model ne öğrenmiş? (duyarlılık analizi)

Tüm özellikler medyanda iken tek bir özelliği ±2 SD değiştirmenin skora etkisi:

| En çok artıran | Δ skor | Yorum |
|---|---|---|
| FiO2 +2 SD | +0.21 | Mantıklı (oksijen ihtiyacı) |
| Direkt bilirubin **−2 SD** | +0.21 | ❌ Fizyolojik olarak anlamsız |
| Laktat **−2 SD** | +0.19 | ❌ Düşük laktat riski artırıyor |
| Hour (YBÜ saati) +2 SD **ve** −2 SD | +0.17 / +0.16 | ❌ Saat iki yönde de riski artırıyor |

Belirgin septik tablo (HR 130, Ateş 38.8, SKB 82, SpO2 88, Laktat 4.5, WBC 16):
**1. saatte %7.4, 24. saatte %8.8, 100. saatte %24.5.**

**Yorum:** Eğitim verisinde laboratuvar değerlerinin %88–99'u eksik ve eksikler medyanla dolduruluyor.
Model, değerin *ne olduğunu* değil, **medyandan farklı olup olmadığını** (yani “ölçülmüş mü”) öğrenmiş.
Ayrıca pozitif pencerelerin medyan YBÜ saati 42, negatiflerin 22; model kalış süresine dayanıyor.
Vital bulgular skoru neredeyse hiç değiştirmiyor.

---

## 5. Yapılması gerekenler (öncelik sırasıyla)

### 🔴 Kritik — model

1. **Eksik veri stratejisini değiştir.** Hasta içinde ileri doldurma (forward-fill), ardından medyan;
   her özellik için **eksiklik maskesi** (0/1) ve **son ölçümden bu yana geçen süre** özellikleri ekle.
   Literatürde PhysioNet 2019 için standart yaklaşım budur. Böylece “ölçülmüş mü” bilgisi açıkça verilir ve değerden ayrılır.
2. **Hour/ICULOS'un etkisini kontrol et.** Ya çıkar ya da kalış süresine göre tabakalı değerlendirme yap.
   Aksi halde model “uzun yatan hasta = sepsis” kısayolunu öğreniyor.
3. **Klinik tutarlılık testleri ekle** (`tests/test_clinical_sanity.py`): ateş/taşikardi/hipotansiyon/laktat
   arttıkça skor azalmamalı (monotonluk). Yeni her model bu testten geçmeli.
4. **Güçlü bir temel model ile karşılaştır.** Aynı özelliklerle LightGBM/XGBoost (PhysioNet 2019
   kazananları ağaç tabanlıydı) ve klinik skorlar (qSOFA, NEWS2, SIRS). GRU bunları geçmiyorsa raporda belirtilmeli.
5. **Değerlendirme metriğini zenginleştir:** hasta düzeyinde metrikler (sepsis başlangıcından kaç saat önce alarm),
   PhysioNet Utility Score, GroupKFold ile güven aralıkları.
6. **Eşik seçimini klinik hedefe göre yap** (ör. duyarlılık ≥ %80) ve raporda yanlış alarm yükünü
   “hasta başına günlük alarm” olarak ver.

### 🟡 Önemli — tutarlılık ve tekrar üretilebilirlik

7. `data/processed/*.pkl` dosyaları commit edilmemiş değişiklik içeriyor (41 → 40 özellik). Veri sürümünü,
   model dosyasını ve preprocessing nesnelerini **birlikte** sürümle (ör. `models/optimized/` altında `column_info.pkl` kopyası + hash).
8. Raporlar (`TUBITAK_Proje_Raporu.md`, `FINAL_DOKUMANTASYON.md`, README) aktif modeli “Bi-GRU + Attention, AUC 0.88”
   olarak anlatıyorsa güncellenmeli. Aktif model attention içermiyor (Keras Tuner `use_attention=False` seçti).
9. `X_*.npy` float64 (≈2.6 GB). float32'ye çevirmek bellek ve süreyi yarıya indirir.
10. Kök dizindeki çok sayıda `train_*.py` / `evaluate_*.py` dosyası ve kullanılmayan modeller (`dual_head` boş,
    `gru_v24_56features`) `experiments/` altında toplanmalı. Hangisinin “resmi” olduğu README'de yazılmalı.
11. `requirements.txt` içinde `flask` ve `flask-cors` yok, sürümler sabitlenmemiş (`requirements_full.txt` daha doğru).
12. `calibrator.pkl` `.gitignore`'daki `*.pkl` kuralına takılır. Repo klonlanırsa `python scripts/fit_calibration.py`
    çalıştırılmalı ya da `git add -f` ile eklenmeli.

### 🟡 Önemli — uygulama ve güvenlik

13. **Kimlik doğrulama demo modunda:** `/api/auth/login` her kullanıcı adı ve şifreyi kabul ediyor; API uçlarında yetki kontrolü yok.
    `auth.py` / `app_auth_endpoints.py` hazır ama bağlanmamış.
14. `CORS(app)` tüm kaynaklara açık. Yalnızca kendi alan adına izin verilmeli.
15. **Denetim izi yok:** kim, hangi hastaya, ne zaman veri girdi kaydedilmiyor.
16. **Otomatik test yok.** En azından API, `reliability.py` ve klinik tutarlılık için pytest eklenmeli.
17. Biyobelirteçler (PCT, CRP, Presepsin, IL-6) eğitimde “şüpheli/sentetik” diye çıkarılmış. Formda artık
    “Modelde kullanılmaz” olarak işaretli. Gerçek veriyle yeniden değerlendirilmeli.
18. SQLite eşzamanlı yazmada yetersiz. Çok kullanıcılı kullanım için PostgreSQL şeması (`database/`) devreye alınmalı.

### 🟢 İyileştirme

19. Hasta düzeyinde PDF rapor, alarm onaylama (acknowledge) akışı.
20. SHAP ile tahmin başına özellik katkısı. `shap` zaten bağımlılıklarda var. Ancak önce madde 1–3 çözülmeli,
    yoksa açıklamalar yanıltıcı olur.

---

## 6. Bu çalışmada yapılan değişiklikler

| Dosya | Değişiklik |
|---|---|
| `reliability.py` (yeni) | Minimum veri kapısı, tamlık, zaman kapsamı, MC-Dropout, OOD, kalibrasyon, model–klinik çelişkisi |
| `scripts/fit_calibration.py` (yeni) | İzotonik kalibratör, ECE/Brier, eşik metrikleri, güvenilirlik diyagramı → `models/optimized/` |
| `app.py` | Boş form reddi; “Yetersiz Veri” durumu; güvenilirlik hesaplama ve DB'ye kaydetme (`reliability`, `assessment` sütunları, otomatik migration); FiO2 % → oran dönüşümü; MAP'in SBP/DBP'den türetilmesi; risk eşikleri ve klinik kurallar artık `config.py`'den; hasta listesinde “son risk” artık **son saatin** riski (önceden `MAX()` ile metin karşılaştırması yapılıyordu); `/api/model-info`; statik dosya servisinde beyaz liste (önceden `patients.db` ve `.env` tarayıcıdan indirilebiliyordu); sabit kodlu `Unit1=1` kaldırıldı |
| `config.py` | Model bilgisi gerçek mimariyle düzeltildi; `SEPSIS_DB_PATH` ortam değişkeni |
| `index.html`, `style.css`, `script.js` | Arayüz yeniden yazıldı (aşağıda) |

### Arayüz

- **Hasta detayı:** solda veri girişi, sağda sabit **Değerlendirme paneli**: model skoru, risk ölçeği üzerinde
  %90 belirsizlik bandı, kalibre olasılık, güvenilirlik halkası, 4 bileşen çubuğu, “Neden?” listesi, tek cümlelik klinik yorum.
- **Form:** 5 adımlı sihirbaz yerine açılır gruplar. Temel vitaller ve kritik lab açık, diğerleri kapalı.
  Canlı veri tamlığı çubuğu, alan bazında aralık uyarısı, kalış saati otomatik (son saat + 1), “Son saati kopyala”.
- **Risk trendi:** belirsizlik bandı ve karar eşiği çizgisi; yetersiz veri saatleri gri.
- **Genel bakış / Alarmlar:** riske göre sıralı liste, güvenilirlik sütunu, “değerlendirilemeyen” ve “düşük güvenilirlik” alarmları.
- **Model & Güvenilirlik sayfası:** sabit kodlu sayılar yerine `/api/model-info`'dan gelen ölçülmüş değerler ve eşik seçenekleri tablosu.
- Açık tema varsayılan, koyu tema (YBÜ gece modu) seçilebilir; mobil uyumlu.

### Davranış değişiklikleri (bilinmesi gerekenler)

- Risk eşikleri artık `config.RISK_LEVELS`'tan geliyor: “Çok Yüksek” 0.50 yerine **0.55**'ten başlıyor, etiketler sadeleşti
  (“Orta Risk (Şüpheli)” → “Orta”). Eski kayıtlar eski etiketleriyle görünür.
- Güvenilirlik özelliğinden önce girilen kayıtlarda güvenilirlik boş görünür; o saat yeniden kaydedilirse hesaplanır.
- Yaş girilmemişse model artık 60 değil eğitim medyanını (64) kullanır ve bu gerekçelerde belirtilir.

---

## 7. Yeni model: GRU v3 (aktif)

### 0.88'in kaynağı

`gru_v23_best` 41 özellik kullanıyordu; 41. özellik `Unnamed: 0` sütunuydu (`Hour` sütununun birebir kopyası,
sızıntı değil). Asıl sorun bölme yöntemiydi: o sürüm `train_test_split` ile **pencereleri rastgele** böldü.
Aynı hastanın bir saati eğitimde, bir sonraki saati testte oldu. Yaş + Cinsiyet + HospAdmTime üçlüsü
hastaların **%93'ünü tekil olarak** tanımladığı için model hastayı tanıyabildi. Kanıt: v23, rastgele seçilen
4.000 hastada (çoğunu eğitimde görmüş) 0.936 AUC alıyor.

### v3'te değişenler (`features.py`, `scripts/train_v3.py`)

| Değişiklik | Neden |
|---|---|
| Hasta içinde ileri doldurma + 34 ölçüm maskesi | Model “ölçülmüş mü” bilgisini değerden ayrı görsün |
| HospAdmTime çıkarıldı, Unit1/Unit2 çıkarıldı | Hasta parmak izi (ilk denemede eğitim AUC 0.92 / val 0.75) |
| Ölçekleyici yalnızca eğitim hastalarıyla fit | Eski pipeline tüm veriyle fit ediyordu (küçük sızıntı) |
| İlk 5 saat de eğitimde (geri kopyalama ile) | Uygulama 1. saatten tahmin yapıyor |
| GaussianNoise + L2 + Dropout 0.4 | Ezberlemeyi azaltmak |
| Hasta bazlı, tabakalı 60/20/20 bölme | Gerçekçi değerlendirme |

### Sonuçlar (8.066 görülmemiş test hastası)

| | Optimized v1 (eski) | **GRU v3** | LightGBM (aynı özellikler) |
|---|---|---|---|
| ROC-AUC | 0.809 | **0.806** | 0.811 |
| PR-AUC | 0.089 | **0.095** | 0.100 |
| ECE (kalibre) | 0.002 | **0.001** | — |
| Normal / septik vital (24. saat) | %5 / %9 | **%1.7 / %41** | — |
| Klinik tutarlılık testi | ❌ | **✅** | — |

Ayrım gücü (AUC) benzer kaldı, ancak model artık **klinik olarak anlamlı** davranıyor: ateş, SpO2, solunum
ve nabız skoru artırıyor; düşük laktat artık riski yükseltmiyor. MAP/SKB'nin etkisi hâlâ zayıf (+0.5 puan),
bu bir sonraki iyileştirme konusu.

### Hâlâ yapılabilecekler

- Daha uzun pencere (12–24 saat) ve “son ölçümden bu yana geçen süre” özellikleri
- MAP/SKB etkisini artırmak için trend özellikleri (6 saatlik değişim)
- GRU + LightGBM topluluğu (ensemble)
- PhysioNet Utility Score ile değerlendirme

### Sunum ve çalıştırma

- `baslat.bat` → tek tıkla başlatma (Python 3.10'u bulur, paketleri kontrol eder, tarayıcıyı açar)
- `sistem_kontrol.bat` → uçtan uca test (geçici veritabanı ile)
- `py -3.10 scripts/demo_hastalari.py` → 3 örnek hasta (`--sil` ile kaldırılır)
- Eski modele dönmek için: `set SEPSIS_MODEL=legacy` ve ardından `py -3.10 app.py`
