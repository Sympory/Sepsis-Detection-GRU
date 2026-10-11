# SepsisAI — Mevcut Durum ve 6 Haftalık Çalışma Planı

Bu belge üç soruyu yanıtlar:
1. **Şu ana kadar ne yapıldı, hangi teknikler kullanıldı, ne sonuç verdi?**
2. **Literatürde benzer çalışmalar ne sonuç alıyor?**
3. **Önümüzdeki 6 haftada ne yapılacak, model yeniden eğitilebilir mi?**

---

## 1. Mevcut durum (ön çalışma — “0. hafta”)

Şu an elimizde **çalışan bir uçtan uca prototip** var. Bu bir son ürün değil, 6 haftalık araştırmanın **başlangıç noktası (baseline)**.

| Bileşen | Durum |
|---------|-------|
| Veri hazırlama (ileri doldurma, maskeler, hasta bazlı bölme) | ✅ Çalışıyor |
| GRU v3 modeli + kalibrasyon | ✅ Eğitildi (GPU'da ~3 dk) |
| Güvenilirlik skoru (MC-Dropout, veri tamlığı) | ✅ Çalışıyor |
| Web arayüzü (Flask + HTML/JS) | ✅ Çalışıyor |
| Klinik tutarlılık testi, LightGBM karşılaştırması | ✅ Var |
| Çapraz doğrulama, Utility Score, açıklanabilirlik, dış doğrulama | ❌ **Plan kapsamında** |

### Kullanılan teknikler

| Alan | Teknik | Neden |
|------|--------|-------|
| **Eksik veri** | Hasta içi ileri doldurma (forward-fill) | Lab değerleri saatlerce geçerli kabul edilir |
| | Ölçüm maskeleri (34 adet 0/1) | “Ölçüldü mü” bilgisi değerden ayrı verilir |
| | Eğitim medyanı ile doldurma | Hiç ölçülmemiş değerler için |
| **Ölçekleme** | Z-skoru (yalnızca eğitim hastalarıyla fit) | Test verisinden bilgi sızmasın |
| **Değerlendirme** | Hasta bazlı, tabakalı 60/20/20 bölme | Aynı hasta iki sette olmasın |
| **Model** | Bidirectional GRU + GRU + Dense | Zaman serisi, az parametre |
| **Düzenlileştirme** | Dropout 0.4, L2 (1e-4), Gaussian gürültü, erken durdurma | Aşırı öğrenmeyi önlemek |
| **Dengesiz veri** | Sınıf ağırlığı (pozitif × 5), PR-AUC ile model seçimi | Sepsis oranı %1.8 |
| **Olasılık** | İzotonik kalibrasyon | Skor = gözlenen oran |
| **Belirsizlik** | MC-Dropout (30 geçiş) | Modelin ne kadar emin olduğu |
| **Güvenlik** | Minimum veri kapısı, klinik kırmızı bayrak kuralları | Eksik veride yanlış güvence vermemek |
| **Karşılaştırma** | LightGBM, klinik tutarlılık testi | Modelin makul davranıp davranmadığı |

---

## 2. Deney geçmişi — daha önce ne denendi, ne sonuç verdi?

| # | Model / Deney | Teknik | ROC-AUC | PR-AUC | Ders |
|---|---------------|--------|---------|--------|------|
| 1 | GRU v23 | GRU(64), medyan doldurma, **rastgele saat bölmesi** | 0.880 | 0.180 | ⚠️ İyimser: aynı hasta hem eğitimde hem testte |
| 2 | GRU v24 (56 özellik) | + PCT, CRP, Presepsin vb. 22 biyobelirteç | **0.9997** | 0.999 | ❌ **Sızıntı**: biyobelirteçler sentetik, etiketi ele veriyor. Çıkarıldı |
| 3 | GRU v25 | Focal loss (γ=2, α=0.25) | 0.812 | 0.085 | Hasta bazlı bölme ile gerçekçi değer; olasılıklar şişkin |
| 4 | Optimized v1 | Keras Tuner ile Bi-GRU (attention denendi, seçilmedi) | 0.809 | 0.089 | Septik hastaya %7: **“ölçülmüş mü” kısayolu** |
| 5 | Paper model | Bi-GRU + Attention + Masking + L2 | 0.801 | 0.093 | Attention kazanç getirmedi |
| 6 | Enhanced v2 | Bi-GRU(96) + Bi-GRU(64) + çok başlı attention + 12 türetilmiş özellik | 0.804 | 0.086 | 262 bin parametre, 92 dk eğitim, kazanç yok |
| 7 | MATLAB Bi-LSTM | Bi-LSTM(96) + Bi-LSTM(64) + self-attention | 0.806 | 0.089 | LSTM de aynı bantta |
| 8 | v3 ilk deneme | İleri doldurma + maskeler + HospAdmTime | 0.808 | 0.106 | Eğitim 0.92 / doğrulama 0.75: **hasta ezberleme** |
| 9 | **GRU v3 (aktif)** | HospAdmTime çıkarıldı, gürültü + L2 | **0.806** | **0.095** | ✅ Klinik tutarlılık testini geçen ilk model |
| 10 | LightGBM | Aynı özellikler, son saat + 6 saatlik değişim | 0.811 | 0.100 | Ağaç modeli GRU ile başa baş |

**Ana bulgu:** 3–10 numaralı deneylerde 7 farklı mimari (GRU, Bi-GRU, attention, LSTM, LightGBM, MATLAB)
hasta bazlı değerlendirmede **0.80–0.81 bandında toplanıyor**. Mimari değiştirmek tek başına yetmiyor.
Sınır **veride ve problem tanımında**. 6 haftalık plan bu yüzden önce değerlendirme ve özelliklere odaklanıyor.

### Neden 0.80'de takılıyoruz? (hipotezler — plan bunları test eder)

1. **Pencere kısa:** 6 saat, sepsise giden yavaş kötüleşmeyi görmek için yetersiz olabilir (12–24 saat denenecek).
2. **Trend bilgisi zayıf:** Model “6 saattir düşen tansiyon” gibi eğilimleri ham değerlerden kendisi çıkarmak zorunda.
3. **Ölçüm zamanı bilgisi yok:** “Laktat 10 saat önce ölçülmüş” ile “şimdi ölçülmüş” ayırt edilmiyor.
4. **Etiket gürültüsü:** Sepsis etiketi bir kural (Sepsis-3) ile otomatik konmuş; bazı “negatifler” tanı almamış vakalar olabilir.
5. **Değerlendirme saatlik:** Saatlik AUC, klinik faydayı tam yansıtmaz (erken uyarı ne kadar erken?).

---

## 3. Literatürde ortalama sonuçlar

> Sayılar farklı bölme yöntemleri, etiket tanımları ve test setleri kullanır. Bu yüzden **doğrudan karşılaştırılamaz**, yön göstericidir.

| Kaynak / çalışma türü | Metrik | Tipik değer |
|-----------------------|--------|-------------|
| PhysioNet 2019 Challenge — kazanan takım (Reyna ve ark., 2020) | Normalized Utility (gizli test seti) | **0.36** |
| Challenge'daki takımların çoğu | Utility | 0.2 – 0.35 civarı |
| Bu veriyle yapılan saatlik tahmin çalışmaları (hasta bazlı) | ROC-AUC | yaklaşık **0.80 – 0.85** |
| Rastgele (saat bazlı) bölme kullanan çalışmalar | ROC-AUC | 0.90+ (iyimser) |
| Klinik skorlar (qSOFA, SIRS) | ROC-AUC | genellikle 0.6 – 0.75 |

**Yorum:**
- Challenge'da en başarılı yöntemlerin önemli bir kısmı **gradient boosting** (XGBoost, LightGBM) ve elle türetilmiş zaman özellikleri kullandı. Derin öğrenme modelleri tek başına bunları geçemedi. Bizim LightGBM ≈ GRU sonucumuz bununla tutarlı.
- Kazanan yaklaşımlardan biri (Morrill ve ark.) zaman serisinden **signature / pencere istatistikleri** çıkarıp gradient boosting'e verdi.
- Bizim 0.806 değerimiz literatür bandının **alt ucunda**. Hedef, 6 haftada **0.83–0.85 ROC-AUC** ve **Utility ≥ 0.30**.

---

## 4. Model yeniden eğitilebilir mi?

**Evet, ve kolay.** Tüm eğitim tek komutla, GPU'da yaklaşık 3 dakika sürüyor:

```bash
py -3.10 scripts/train_v3.py --epochs 25 --pos-weight 5 --output models/v3_deneme
py -3.10 scripts/baseline_lightgbm.py
```

Çıktılar (`models/<klasör>/`): model, kalibratör, test metrikleri, klinik tutarlılık testi, ROC/PR/kalibrasyon grafikleri.
Uygulama `config.py` üzerinden hangi modeli kullanacağını seçiyor. Yeni model eskisini bozmadan yan yana denenebilir.

Eğitim süresinin kısa olması, 6 hafta içinde **yüzlerce deney** yapılabileceği anlamına geliyor.

---

## 5. 6 Haftalık Plan

### Hafta 1 — Sağlam değerlendirme altyapısı
*“Neyi ölçtüğümüzden emin olmadan modeli iyileştiremeyiz.”*

| Görev | Teknik |
|-------|--------|
| 5 katlı **GroupKFold** çapraz doğrulama | Hasta bazlı, her sonuç ± güven aralığı ile |
| **PhysioNet Utility Score** uygulaması | Challenge'ın resmi metriği: erken uyarıyı ödüllendirir, geç/yanlış alarmı cezalandırır |
| Hasta düzeyi metrikler | Sepsisten kaç saat önce alarm, hasta başına günlük alarm sayısı |
| Klinik temel modeller | qSOFA, SIRS, NEWS2 skorlarını hesaplayıp karşılaştırmak |
| Deney takibi | Her deneyin ayarları ve sonuçları tek tabloda (CSV/MLflow) |

**Teslim:** `evaluate.py` + tüm mevcut modellerin aynı protokolle yeniden ölçüldüğü tablo.
**Başarı ölçütü:** v3'ün AUC'si ± güven aralığıyla raporlanmış olmalı. qSOFA'nın üzerinde olduğu gösterilmeli.

### Hafta 2 — Özellik mühendisliği
*“Hipotez 2 ve 3: model eğilimi ve ölçüm zamanını görmüyor.”*

| Görev | Teknik |
|-------|--------|
| Ölçümden bu yana geçen süre | Her değişken için “son ölçüm kaç saat önce” (GRU-D fikri) |
| Kayan pencere istatistikleri | 6/12/24 saatlik ortalama, min, maks, std, eğim |
| Klinik türetilmiş skorlar | Şok indeksi (HR/SKB), SOFA alt skorları, qSOFA, PaO2/FiO2 |
| Değişim özellikleri | Δ (şimdi − 1 saat önce), Δ6 (şimdi − 6 saat önce) |
| Ablasyon | Her özellik grubunu çıkarıp etkisini ölçmek |

**Teslim:** `features_v4.py`, ablasyon tablosu.
**Beklenen kazanç:** Literatürde en çok fark yaratan adım bu. Hedef +0.01–0.03 AUC.

### Hafta 3 — Mimari deneyleri
*“Hipotez 1: pencere kısa. Hangi mimari eksik veriyle en iyi başa çıkıyor?”*

| Görev | Teknik |
|-------|--------|
| Pencere uzunluğu | 6 / 12 / 24 / 48 saat |
| **GRU-D** | Eksik veriyi zamanla “sönümleyen” GRU (Che ve ark., 2018) |
| **TCN** | Zamansal evrişim ağı; uzun pencerede hızlı |
| **Transformer encoder** | Self-attention ile uzun bağımlılıklar |
| Hiperparametre arama | Optuna / Keras Tuner, **GroupKFold** içinde (sızıntısız) |

**Teslim:** Mimari karşılaştırma tablosu (AUC, PR-AUC, Utility, parametre, eğitim süresi).

### Hafta 4 — Dengesiz veri, kayıp fonksiyonu ve topluluk
| Görev | Teknik |
|-------|--------|
| Kayıp fonksiyonları | Ağırlıklı BCE, focal loss, **Utility'ye uyumlu kayıp** |
| Örnekleme | Hasta düzeyinde dengeleme, sepsis öncesi saatleri artırma |
| **Topluluk (ensemble)** | GRU + LightGBM ortalaması / stacking |
| Kalibrasyon karşılaştırması | Platt scaling vs izotonik vs temperature scaling |
| Eşik optimizasyonu | Utility'yi maksimize eden eşik |

**Teslim:** En iyi tek model ve en iyi topluluk.
**Beklenen kazanç:** Topluluk genelde tek modelden +0.01–0.02 AUC.

### Hafta 5 — Açıklanabilirlik ve güvenilirlik
*“Doktor neden bu skoru verdiğini sorarsa?”*

| Görev | Teknik |
|-------|--------|
| Tahmin başına açıklama | **SHAP** (LightGBM için TreeSHAP), Integrated Gradients (GRU için) |
| Zaman açıklaması | Hangi saatin, hangi değişkenin skoru artırdığı |
| Belirsizlik karşılaştırması | MC-Dropout vs **Deep Ensembles** (5 model) |
| **Conformal prediction** | Garanti kapsama oranlı tahmin aralıkları |
| Dağılım dışı tespiti | Mahalanobis uzaklığı / autoencoder |
| Arayüze ekleme | “Bu skoru en çok artıran 3 faktör” paneli |

**Teslim:** Arayüzde açıklama paneli, belirsizlik yöntemleri karşılaştırma raporu.

### Hafta 6 — Dış doğrulama, uygulama ve rapor
| Görev | Teknik |
|-------|--------|
| **Dış doğrulama** | PhysioNet verisi iki hastaneden (A ve B). A'da eğitip B'de test: model başka hastaneye genelleniyor mu? |
| Alt grup analizi | Yaş grupları, cinsiyet, YBÜ tipi bazında performans (adalet) |
| Uygulama | Kimlik doğrulama, birim testleri (pytest), Docker |
| Rapor ve sunum | Yöntem, sonuçlar, sınırlılıklar |

**Teslim:** Final rapor, güncellenmiş uygulama, sunum.

---

## 6. Hedefler

| Metrik | Şu an (v3) | 6 hafta hedefi | Literatür |
|--------|-----------|----------------|-----------|
| ROC-AUC (hasta bazlı) | 0.806 | **0.83 – 0.85** | ~0.80 – 0.85 |
| PR-AUC | 0.095 | **0.12+** | — |
| Utility Score | ölçülmedi | **≥ 0.30** | kazanan 0.36 |
| Duyarlılık (yanlış alarm ≤ %10 iken) | %26 | **%40+** | — |
| Açıklanabilirlik | yok | tahmin başına SHAP | — |
| Dış doğrulama | yok | A → B testi | — |

---

## 7. Riskler ve önlemler

| Risk | Önlem |
|------|-------|
| Yeni özellikler sızıntı getirebilir | Her deney GroupKFold + klinik tutarlılık testinden geçer |
| Hiperparametre aramasında test setine aşırı uyum | Test seti yalnızca **6. haftada bir kez** kullanılır |
| Derin model ağaç modelini geçemeyebilir | Bu da geçerli bir sonuç; topluluk yaklaşımı ve açıklanabilirlik katkısı raporlanır |
| GPU / süre kısıtı | Eğitim ~3 dk; ağır deneyler (Transformer, 48 saat pencere) alt örneklemle |

---

## Kaynaklar

- Reyna MA, ve ark. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine* 48(2):210–217.
- Singer M, ve ark. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*.
- Che Z, ve ark. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Scientific Reports*. (GRU-D)
- Morrill J, ve ark. (2019). The Signature-based Model for Early Detection of Sepsis from Electronic Health Records in the Intensive Care Unit. *Computing in Cardiology*.
- Gal Y, Ghahramani Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Lundberg SM, Lee SI. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*. (SHAP)
- Bai S, ve ark. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*. (TCN)
