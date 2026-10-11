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

Bu proje, yoğun bakım ünitesindeki (YBÜ) hastaların saatlik vital bulgu ve laboratuvar
verilerinden **sepsis riskini** tahmin eden bir **GRU (Gated Recurrent Unit)** sinir ağı ve
bu modeli kullanan bir **Flask web uygulaması** içerir.

Model son **6 saatlik** hasta verisine bakar ve sepsis olasılığını üretir. Uygulama her tahmin için
ayrıca **bu tahminin ne kadar güvenilir olduğunu** hesaplar ve gerekçeleriyle birlikte gösterir.

### 🎯 Öne Çıkanlar

- **40.336 hasta, 1,55 milyon saatlik kayıt** — PhysioNet / Computing in Cardiology Challenge 2019 verisi
- **Hasta bazlı test** — test hastalarının hiçbiri eğitimde görülmedi (veri sızıntısı yok)
- **ROC-AUC 0.806, PR-AUC 0.095** — 8.066 görülmemiş test hastasında
- **Klinik tutarlılık** — aynı saatte normal vitallere %1.7, septik vitallere %41 risk
- **Güvenilirlik skoru** — MC-Dropout belirsizliği, veri tamlığı, izotonik kalibrasyon
- **Tek tıkla çalışır** — `baslat.bat` Python'u bulur, paketleri kontrol eder, tarayıcıyı açar

---

## 🚀 Hızlı Başlangıç

### Windows (önerilen)

1. **`baslat.bat`** dosyasına çift tıklayın.
2. ~20 saniye sonra tarayıcıda `http://localhost:5000` açılır.

Sunumdan önce her şeyin çalıştığını doğrulamak için **`sistem_kontrol.bat`**:

```
=== SepsisAI sistem kontrolü ===
  [OK] Python sürümü — 3.10
  [OK] Aktif model — v3 → models\v3\model.keras
  [OK] Boş form reddediliyor
  [OK] Tek vital → Yetersiz Veri
  [OK] Normal hasta tahmini — skor %2.5, Çok Düşük
  [OK] Septik hasta tahmini — skor %37.0, Çok Yüksek
  [OK] Model septik hastaya daha yüksek skor veriyor
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

✅ **Zaman serisi modelleme**: 6 saatlik geriye bakış penceresi, iki katmanlı GRU
✅ **71 özellik**: 34 klinik değişken + 34 “bu saat ölçüldü mü” maskesi + yaş, cinsiyet, YBÜ saati
✅ **İleri doldurma**: Laboratuvar değerleri her saat ölçülmez; son ölçülen değer geçerli kabul edilir
✅ **Dengesiz veri**: Pozitif oran %1.8; sınıf ağırlığı + PR-AUC ile erken durdurma
✅ **Kalibrasyon**: İzotonik regresyon. Ham skor, gözlenen sepsis oranına çevrilir

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

Tüm değerler, eğitimde **hiç görülmemiş 8.066 test hastası** (308.743 saatlik pencere, %1.8 pozitif) üzerindedir.

| Metrik | Değer | Yorum |
|--------|-------|-------|
| **ROC-AUC** | 0.806 | Sepsis/sepsis değil ayrımı |
| **PR-AUC** | 0.095 | Rastgele tahminin (0.018) ~5 katı |
| **Brier (kalibre)** | 0.017 | Olasılıkların doğruluğu (düşük = iyi) |
| **ECE (ham → kalibre)** | 0.040 → 0.001 | Kalibrasyon hatası |

### Karşılaştırma (aynı test hastaları, aynı özellikler)

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **GRU v3** (bu proje) | 0.806 | 0.095 |
| LightGBM (son saat + 6 saatlik değişim) | 0.811 | 0.100 |

İki model neredeyse aynı performansta. Bu veri setinde (PhysioNet 2019) ağaç tabanlı modellerin
güçlü olduğu bilinir. GRU'nun avantajı zaman serisini doğrudan işlemesi, MC-Dropout ile belirsizlik
verebilmesi ve tek bir uçtan uca modelle çalışmasıdır.

### İki çalışma noktası

Sepsis nadir olduğu için tek bir “doğru” eşik yoktur. Seçim, kaçırılan vaka ile yanlış alarm arasındaki dengedir:

| Eşik | Duyarlılık | Özgüllük | PPV | NPV | Kullanım |
|------|-----------|----------|-----|-----|----------|
| **0.35** (varsayılan, F1) | %25.6 | %96.7 | %12.6 | %98.6 | Az alarm, çok vaka kaçar |
| **0.032** (tarama) | %80.9 | %63.9 | %4.0 | %99.5 | Az vaka kaçar, çok alarm |

### Karışıklık matrisi (eşik 0.35)

|  | Gerçek: Sepsis | Gerçek: Sepsis değil |
|---|---|---|
| **Tahmin: Sepsis** | 1.425 (TP) | 9.883 (FP) |
| **Tahmin: Sepsis değil** | 4.147 (FN) | 293.288 (TN) |

### Klinik tutarlılık testi

Aynı YBÜ saatinde iki hayali hasta: normal vitaller (HR 78, Ateş 36.8, SKB 125) ve septik tablo
(HR 130, Ateş 38.9, SKB 82, SpO2 88, Laktat 4.5, WBC 17).

| YBÜ saati | Normal | Septik | Eski model (septik) |
|-----------|--------|--------|---------------------|
| 6 | %2.3 | **%37.1** | %7.6 |
| 24 | %1.7 | **%41.4** | %8.8 |
| 72 | %14.9 | **%63.3** | — |

Grafikler: `models/v3/evaluation.png` (ROC, PR, güvenilirlik diyagramı) ve `models/v3/training_history.png`.

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

## 🌐 Web Uygulaması

### API

| Metot | Uç nokta | Açıklama |
|-------|----------|----------|
| GET | `/` | Arayüz |
| GET | `/api/patients` | Hastalar ve son risk/güvenilirlik |
| POST | `/api/patients` | Yeni hasta |
| GET | `/api/patients/<id>` | Hasta detayı ve saatlik kayıtlar |
| POST | `/api/patients/<id>/hourly-data` | Saatlik veri ekle → tahmin + güvenilirlik |
| DELETE | `/api/patients/<id>` | Hasta sil |
| GET | `/api/model-info` | Aktif model, metrikler, kalibrasyon istatistikleri |
| GET | `/api/health` | Sistem durumu |

Örnek yanıt (demo hasta “Mehmet Kaya”, YBÜ 24. saat):

```json
{
  "prediction": 0.376,
  "risk_level": "Çok Yüksek",
  "is_sepsis_risk": true,
  "threshold": 0.35,
  "reliability": {
    "score": 72.7,
    "level": "Yüksek",
    "components": {"completeness": 87.5, "temporal": 100.0, "certainty": 13.3, "in_distribution": 100.0},
    "mc_dropout": {"mean": 0.371, "p05": 0.263, "p95": 0.481, "agreement": 0.567, "n": 30},
    "calibrated_probability": 0.095,
    "reasons": [
      "Son 6 saatte ölçülmemiş kritik lab: Total Bilirubin",
      "Model kararsız: stokastik geçişlerin %57'i aynı kararı verdi (skor aralığı %26–%48)."
    ]
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

## 🏗️ Mimari

### Sistem

```
Tarayıcı (index.html, script.js, style.css)
        │  JSON / REST
        ▼
Flask API (app.py) ──── SQLite (patients.db)
        │
        ├── features.py      → ileri doldurma, ölçüm maskeleri, ölçekleme
        ├── GRU modeli       → models/v3/model.keras
        └── reliability.py   → MC-Dropout, tamlık, kalibrasyon, güvenilirlik skoru
```

### Model (v3)

```
Girdi: (batch, 6 saat, 71 özellik)
    ↓
┌───────────────────────────────────┐
│  GaussianNoise(0.1)               │  ← yalnızca eğitimde, ezberlemeyi azaltır
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Bidirectional GRU(64) → 128      │  ← zaman içindeki değişim, iki yönlü
│  L2(1e-4) + Dropout(0.4)          │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  GRU(64) + Dropout(0.4)           │  ← pencerenin özeti
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Dense(32, ReLU) + Dropout(0.4)   │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Dense(1, Sigmoid)                │
└───────────────────────────────────┘
    ↓
Çıktı: P(Sepsis) ∈ [0, 1]  →  izotonik kalibrasyon
```

**Parametre sayısı:** 91.969

---

## 🔬 Yöntem

### Veri hazırlama (`features.py`)

1. **Temizlik**: Fizyolojik olarak imkânsız değerler silinir (ör. HR > 300, Ateş < 25)
2. **Ölçüm maskesi**: Her değişken için “bu saat ölçüldü mü” (0/1)
3. **İleri doldurma**: Hasta içinde son ölçülen değer taşınır
4. **Medyan doldurma + standartlaştırma**: Yalnızca **eğitim hastalarıyla** fit edilir
5. **Kayan pencere**: Her saat için son 6 saat; ilk saatlerde ilk ölçüm geriye kopyalanır
6. **Bölme**: Hasta bazlı, sepsis durumuna göre tabakalı %60 / %20 / %20

### Eğitim (`scripts/train_v3.py`)

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
| **v3** | 0.806 | — | Klinik tutarlılık testini geçiyor, kalibre |

**Ana ders:** Daha yüksek AUC her zaman daha iyi model demek değildir. 0.88'lik model, görmediği
hastada 0.81'lik modelden daha iyi değildi. Doğru değerlendirme yöntemi metrik kadar önemlidir.

---

## 🗺️ Yol Haritası (6 hafta)

Mevcut sistem çalışan bir **başlangıç noktasıdır**. 7 farklı mimari (GRU, LSTM, attention, LightGBM…) hasta bazlı
değerlendirmede 0.80–0.81'de toplanıyor. Bu yüzden plan önce değerlendirme ve özelliklere odaklanır:

| Hafta | Konu | Ana teknikler |
|-------|------|---------------|
| 1 | Değerlendirme altyapısı | GroupKFold, PhysioNet Utility Score, qSOFA/NEWS2 karşılaştırması |
| 2 | Özellik mühendisliği | Ölçümden geçen süre, 6/12/24 saatlik trendler, SOFA, şok indeksi |
| 3 | Mimari | 12–48 saatlik pencere, GRU-D, TCN, Transformer, Optuna |
| 4 | Kayıp ve topluluk | Focal / Utility kaybı, GRU + LightGBM ensemble, kalibrasyon |
| 5 | Açıklanabilirlik | SHAP, Integrated Gradients, Deep Ensembles, conformal prediction |
| 6 | Dış doğrulama ve rapor | Hastane A → B testi, alt grup analizi, pytest, Docker |

**Hedef:** ROC-AUC 0.83–0.85, Utility ≥ 0.30 (PhysioNet 2019 kazananı: 0.36).

Ayrıntılar: [docs/PROJE_PLANI_6_HAFTA.md](docs/PROJE_PLANI_6_HAFTA.md) · Mimari açıklaması: [docs/MIMARI_ACIKLAMA.md](docs/MIMARI_ACIKLAMA.md)

---

## 📁 Proje Yapısı

```
├── baslat.bat                  # Tek tıkla başlatma (Windows)
├── sistem_kontrol.bat / .py    # Uçtan uca test
├── app.py                      # Flask API + tahmin
├── config.py                   # Model seçimi, eşikler, risk seviyeleri
├── features.py                 # v3 özellik mühendisliği (eğitim ve uygulama ortak)
├── reliability.py              # Güvenilirlik skoru, MC-Dropout
├── index.html / script.js / style.css
│
├── scripts/
│   ├── train_v3.py             # v3 eğitimi + kalibrasyon + değerlendirme
│   ├── baseline_lightgbm.py    # Karşılaştırma modeli
│   ├── fit_calibration.py      # Eski model için kalibrasyon
│   └── demo_hastalari.py       # Sunum için örnek hastalar
│
├── models/
│   ├── v3/                     # Aktif model, dönüştürücü, kalibratör, grafikler
│   └── optimized/              # Eski model (SEPSIS_MODEL=legacy ile seçilir)
│
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
