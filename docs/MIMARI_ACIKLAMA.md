# Model Mimarisi — Satır Satır Açıklama

Aktif model: **v4 hibrit** (sinir ağı + LightGBM). Önceki model **GRU v3** ikinci bölümde anlatılıyor; ödevde
"önce GRU ile başladık, özellik mühendisliği ve hibrit modelle 0.806 → 0.847'ye çıktık" hikâyesi için ikisi de lazım.

---

# Bölüm 1 — v4 Hibrit Model (aktif)

```
1. Hastanın yatıştan bu yana saatlik verisi → 281 zaman serisi özelliği (son saat için)
2. Dal A — LightGBM: 3 seed × ~185 ağaç (toplam 556), saf NumPy ile çalışır
3. Dal B — Sinir ağı:
     GaussianNoise(0.1) — yalnızca eğitimde
     Dense(256, ReLU) → BatchNorm → Dropout(0.5)
     Dense(128, ReLU) → BatchNorm → Dropout(0.5)
     Dense(64, ReLU)  → Dropout(0.5)
     Dense(1, Sigmoid)
4. Birleşim: 0.7 × LightGBM + 0.3 × sinir ağı
5. İzotonik kalibrasyon → P(Sepsis); MC-Dropout (sinir ağı dalı) → belirsizlik aralığı
```

Kod: `features_v4.py` (özellikler), `scripts/train_hybrid.py` (eğitim), `ml_service.py` (tahmin), `gbm_numpy.py` (ağaçlar)

## 1. Girdi: 281 zaman serisi özelliği

GRU v3 modele son 6 saatin ham değerlerini veriyordu; "nabız 24 saattir yükseliyor" gibi bir eğilimi model kendisi
çıkarmak zorundaydı. v4'te bu bilgiler **açıkça hesaplanıp** verilir. Her özellik yalnızca o saate kadarki veriden
hesaplanır (geleceğe bakmaz).

| Grup | Sayı | Örnek |
|------|------|-------|
| Son bilinen değer (ileri doldurma) | 34 | Nabız = 118 |
| Bu saat ölçüldü mü | 34 | Laktat bu saat ölçüldü: 1 |
| Son ölçümden geçen saat | 34 | Son FiO2 ölçümü 5 saat önce |
| Şimdiye kadarki ölçüm sayısı | 34 | 12 kez SpO2 ölçüldü |
| 6 ve 24 saatlik ortalama, min, maks (+ vitallerde std) | 84 | 6 saatlik en yüksek ateş |
| Değişim (1 ve 6 saat önceye göre) | 24 | Nabız 6 saatte +25 |
| Yatıştan bu yana en yüksek/düşük/ortalama, ortalamadan sapma | 24 | Yatıştan bu yana en düşük tansiyon |
| Klinik skorlar | 9 | Şok indeksi (nabız/tansiyon), SIRS, kısmi SOFA ve 24 saatlik değişimi, SpO2/FiO2, nabız basıncı, BUN/kreatinin |
| Yaş, cinsiyet, YBÜ saati, kayıt sayısı | 4 | |
| **Toplam** | **281** | |

Girilmeyen saatler boş satır olarak eklenir (eğitim verisinde her satır bir saat); böylece "24 saatlik ortalama"
gerçekten 24 saati kapsar.

## 2. Dal A — LightGBM (gradient boosting)

**Karar ağacı:** "Ateş > 38.4 mü? → Evet → Laktat ölçüleli 3 saatten az mı? → …" gibi sorularla veriyi bölen yapı.
**Gradient boosting:** Ağaçlar sırayla eğitilir; her yeni ağaç öncekilerin **hatasını** düzeltmeye çalışır.
Sonuç: yüzlerce küçük ağacın toplamı.

| Ayar | Değer | Anlamı |
|------|-------|--------|
| num_leaves = 31 | | Her ağaçta en fazla 31 yaprak; küçük ağaç = ezberleme riski az |
| min_child_samples = 1000 | | Bir yaprağa en az 1000 örnek düşmeli; tek hastaya özel kural öğrenilemez |
| lambda_l2 = 20 | | Yaprak değerlerine L2 cezası (düzenlileştirme) |
| feature_fraction = 0.4 | | Her ağaç özelliklerin rastgele %40'ını görür |
| learning_rate = 0.03 | | Her ağacın katkısı küçük tutulur |
| scale_pos_weight = 5 | | Sepsis örnekleri 5 kat ağırlıklı (%1.8 pozitif) |
| 3 seed | | Farklı rastgele başlangıçla 3 model, ortalamaları alınır (daha kararlı) |

**Neden saf NumPy?** Windows'ta LightGBM'in C kütüphanesi, TensorFlow ile aynı programda bazı fonksiyonlarda
çöküyor. Eğitilmiş ağaçlar JSON'a aktarılıyor ve uygulamada her ağaç NumPy ile dolaşılıyor (`gbm_numpy.py`).
Sonuç LightGBM'in kendi tahminiyle **birebir aynı** (eğitim betiği her seferinde doğruluyor, fark 0.0).

## 3. Dal B — Sinir ağı (çok katmanlı algılayıcı, MLP)

| Katman | Çıktı | Parametre | Anlamı |
|--------|-------|-----------|--------|
| Girdi | 281 | 0 | Standartlaştırılmış özellikler (z-skoru, ±6 ile kırpılmış) |
| GaussianNoise(0.1) | 281 | 0 | Eğitimde girdiye küçük gürültü; ezberlemeyi azaltır |
| Dense(256, ReLU) | 256 | 72.192 | 281×256 ağırlık + 256 bias. ReLU = max(0, x) |
| BatchNormalization | 256 | 1.024 | Her mini-batch'te aktivasyonları normalize eder; eğitimi hızlandırır ve kararlı kılar |
| Dropout(0.5) | 256 | 0 | Nöronların yarısı rastgele kapatılır (eğitimde ve MC-Dropout'ta) |
| Dense(128, ReLU) | 128 | 32.896 | |
| BatchNormalization | 128 | 512 | |
| Dropout(0.5) | 128 | 0 | |
| Dense(64, ReLU) | 64 | 8.256 | |
| Dropout(0.5) | 64 | 0 | |
| Dense(1, Sigmoid) | 1 | 65 | 0–1 arası olasılık |
| **Toplam** | | **114.945** | |

**Neden GRU değil de MLP?** Özellikler hastanın geçmişini zaten özetliyor. Denemelerde GRU'lar 1–2 epoch sonra hastaları
ezberlemeye başladı (eğitim AUC 0.92, validasyon 0.72); MLP aynı özelliklerle daha iyi genelledi (0.829).

**Eğitim:** Adam (lr 1e-3), batch 4096, binary cross-entropy, pozitif sınıf ağırlığı 5, L2 1e-4, validasyon PR-AUC ile
erken durdurma ve öğrenme hızını yarıya indirme.

## 4. Birleşim (ensemble)

`P = 0.7 × P_LightGBM + 0.3 × P_sinir_ağı`

Ağırlık validasyon hastalarında 0.05 adımlarla denenip PR-AUC'yi en yüksek yapan değer olarak seçildi.
İki model farklı hatalar yaptığı için ortalamaları her birinden daha kararlıdır; sinir ağı dalı ayrıca MC-Dropout ile
**belirsizlik** ölçümü sağlar (ağaç modelinin böyle bir özelliği yok).

## 5. Kalibrasyon, eşik, güvenilirlik

- **İzotonik kalibrasyon:** Ham %42 → gözlenen oran %13.7 (sınıf ağırlıkları skoru yukarı iter). ECE 0.048 → 0.002.
- **Eşik 0.33:** validasyonda F1'i en yüksek yapan değer. Tarama eşiği 0.066 (%80 duyarlılık).
- **MC-Dropout:** sinir ağı 30 kez Dropout açık çalıştırılır; `0.7 × LightGBM + 0.3 × her örnek` → %90 aralık.
  Testte model kararlıyken doğruluk %65, kararsızken %44.

## 6. Sonuçlar (test, tek seferlik)

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Hibrit v4 | **0.847** | **0.132** |
| LightGBM dalı | 0.847 | 0.131 |
| Sinir ağı dalı | 0.823 | 0.113 |
| GRU v3 | 0.806 | 0.095 |

**5 katlı çapraz doğrulama:** hibrit 0.842 ± 0.009 (PR-AUC 0.129 ± 0.010), LightGBM 0.840 ± 0.010, MLP 0.820 ± 0.006.

**Klinik skorlarla (test):** NEWS2 kısmi 0.682, SIRS 0.649, qSOFA kısmi 0.593. PhysioNet Utility: hibrit **0.427**, GRU v3 0.358, NEWS2 0.115.
Utility iç test hastalarında ölçüldü; yarışmanın gizli hastane skoruyla (kazanan 0.36) doğrudan karşılaştırılamaz.

---

# Bölüm 2 — GRU v3 (önceki model)


```
1. Input(6 saat × 71 özellik: 34 değer + 34 ölçüm maskesi + yaş, cinsiyet, saat)
2. GaussianNoise(0.1) — yalnızca eğitimde
3. Bidirectional GRU(64) → 128, L2(1e-4)
4. Dropout(0.4)
5. GRU(64), L2(1e-4)
6. Dropout(0.4)
7. Dense(32, ReLU) → Dropout(0.4)
8. Dense(1, Sigmoid) → P(Sepsis)
```

Kod: `scripts/train_v3.py` → `build_model()` · Özellikler: `features.py`

---

## Büyük resim: veri modelin içinden nasıl akar?

| Adım | Katman | Çıktı şekli | Parametre |
|------|--------|-------------|-----------|
| 1 | Input | (6, 71) | 0 |
| 2 | GaussianNoise | (6, 71) | 0 |
| 3 | Bidirectional GRU(64) | (6, 128) | 52.608 |
| 4 | Dropout | (6, 128) | 0 |
| 5 | GRU(64) | (64) | 37.248 |
| 6 | Dropout | (64) | 0 |
| 7 | Dense(32) + Dropout | (32) | 2.080 |
| 8 | Dense(1) | (1) | 33 |
| | **Toplam** | | **91.969** |

Kısaca: **6 saatlik tablo** girer → GRU'lar saatler arasındaki değişimi okur → **64 sayılık bir özet** çıkar →
küçük bir sınıflandırıcı bu özeti **0–1 arası bir olasılığa** çevirir.

---

## 1. `Input(6 saat × 71 özellik)`

Model her tahminde **6 satır × 71 sütunluk bir tablo** görür. Her satır bir saat, en alttaki satır şu an.

**Neden 6 saat?** PhysioNet 2019 verisinde `SepsisLabel`, klinik sepsis başlangıcından **6 saat önce** 1 olur.
Amaç sepsisi 6 saat önceden yakalamak. 6 saatlik pencere, eğilimi görmeye yetecek kadar uzun, ilk saatlerde
tahmin yapabilecek kadar da kısa. Hasta 6 saatten azdır yoğun bakımdaysa ilk satır geriye kopyalanır.

**71 özellik:**

| Grup | Sayı | Açıklama |
|------|------|----------|
| Klinik değerler | 34 | HR, SpO2, ateş, KB, solunum, laktat, WBC, kreatinin… Son ölçülen değer ileri taşınır, sonra **z-skoru**: (x − ortalama) / std |
| Ölçüm maskeleri | 34 | O değişken **bu saat ölçüldü mü?** 1 = ölçüldü, 0 = ileri taşındı veya hiç yok |
| Yaş, cinsiyet | 2 | Statik bilgiler |
| Saat | 1 | Yoğun bakıma girişten bu yana geçen saat |

**Maskeler neden var?** Laboratuvar değerlerinin %90'dan fazlası eksik. Eski modelde eksikler medyanla
dolduruluyordu; model “değer medyandan farklıysa ölçülmüştür, ölçüldüyse hasta ağırdır” kısayolunu öğrendi
ve düşük laktatı bile risk saydı. Maske bu bilgiyi **ayrı ve açık** bir sinyal olarak verir. Böylece değerin
kendisi doğru yorumlanır (Che ve ark., 2018).

**Neden HospAdmTime yok?** Yaş + cinsiyet + hastaneye yatış süresi hastaların %93'ünü tekil olarak tanımlıyordu.
Model bu “parmak izi” ile hastayı ezberledi (eğitim AUC 0.92, doğrulama 0.75). Çıkarınca düzeldi.

---

## 2. `GaussianNoise(0.1)` — yalnızca eğitimde

Eğitim sırasında her girdiye ortalaması 0, standart sapması **0.1** olan rastgele gürültü eklenir.
Girdiler z-skoru olduğu için 0.1, her değerin kendi standart sapmasının %10'u kadar oynaması demektir
(nabız için yaklaşık ±1–2 atım).

- **Amaç:** Düzenlileştirme (regularization). Model tam değerleri ezberleyemez, genel örüntüyü öğrenmek zorunda kalır.
  Ölçüm hatasına karşı da dayanıklı olur.
- **Tahmin sırasında kapalıdır.** Keras bu katmanı sadece `training=True` iken çalıştırır.

---

## 3. `Bidirectional GRU(64) → 128, L2(1e-4)`

### GRU nedir?
**Gated Recurrent Unit**, zaman serisi için tekrarlayan (recurrent) bir sinir ağı hücresidir.
Saatleri sırayla okur ve her adımda 64 sayılık bir **gizli durum** (hafıza) `h` günceller:

```
z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)          ← güncelleme kapısı: eski hafızanın ne kadarı kalsın?
r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)          ← sıfırlama kapısı: yeni bilgi hesaplanırken geçmiş ne kadar kullanılsın?
h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h) ← aday yeni hafıza
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t          ← eski ve yeni hafızanın karışımı
```

Kapılar sayesinde model “ateş 3 saattir yükseliyor” gibi bilgiyi hafızada tutabilir.
LSTM'e göre bir kapı eksiktir (3'e karşı 2 kapı), dolayısıyla **daha az parametreli ve daha hızlıdır**,
bu boyutta veride benzer sonuç verir (Chung ve ark., 2014).

### 64 ne demek?
Gizli durumun boyutu, yani modelin her saat için tuttuğu **hafıza vektörünün uzunluğu**.
Daha büyük olursa model daha karmaşık örüntü öğrenebilir ama ezberleme riski artar.

### Bidirectional ve 128
Pencere iki yönde okunur:
- **İleri:** 1. saat → 6. saat (“nasıl kötüleşti?”)
- **Geri:** 6. saat → 1. saat (“şu anki durum geçmişte nereden geldi?”)

İki yönün 64'er sayılık çıktısı yan yana eklenir: **64 + 64 = 128**. `return_sequences=True` olduğu için
her saat için ayrı bir 128'lik vektör çıkar, sonuç (6, 128) olur.

> **Bu gelecek bilgisi sızdırmaz mı?** Hayır. Pencerenin 6 saatinin hepsi tahmin anında **zaten ölçülmüş** geçmiş
> verilerdir. Geri yönlü okuma, yalnızca bu geçmişi ters sırada işler.

### L2(1e-4)
Kayıp fonksiyonuna `0.0001 × Σ(ağırlık²)` eklenir. Büyük ağırlıklar cezalandırılır, model daha yumuşak ve
genelleyebilen bir çözüm seçer (**weight decay**).

### Parametre sayısı (52.608)
Her yön için 3 kapı × (girdi ağırlıkları 71×64 + hafıza ağırlıkları 64×64 + 2×64 bias) = 26.304, iki yön × 26.304 = **52.608**.

---

## 4. `Dropout(0.4)`

Eğitimde her adımda nöron çıktılarının rastgele **%40'ı sıfırlanır**, kalanlar 1/0.6 ile büyütülür.
Model tek bir nörona güvenemez, bilgiyi dağıtmak zorunda kalır. Bu bir **düzenlileştirme** yöntemidir
(Srivastava ve ark., 2014).

**Bu projede ikinci bir görevi var: MC-Dropout.** Tahmin sırasında Dropout'u bilerek açık bırakıp modeli
**30 kez** çalıştırıyoruz. Her seferinde farklı nöronlar kapandığı için 30 farklı skor çıkar:
- Skorlar birbirine yakınsa model **emin**,
- dağınıksa model **kararsız**.

Bu, uygulamadaki **%90 belirsizlik aralığını** ve “model kararlılığı” bileşenini üretir (Gal ve Ghahramani, 2016).

---

## 5. `GRU(64), L2(1e-4)`

İkinci GRU katmanı, ilk katmanın her saat için ürettiği 128'lik vektörleri okur.
`return_sequences=False` olduğu için **yalnızca son saatteki** gizli durumu verir. Bu, 6 saatin tamamının
**64 sayılık özetidir**.

- **Neden iki katman?** İlk katman saat saat yerel değişimleri, ikincisi bunların birleşimini
  (“ateş + taşikardi + düşen tansiyon birlikte”) öğrenir.
- **Parametre:** 3 × (128×64 + 64×64 + 2×64) = **37.248**

---

## 6. `Dropout(0.4)`

4. adımdakiyle aynı: 64'lük özetin %40'ı eğitimde rastgele sıfırlanır, MC-Dropout'ta da kullanılır.

---

## 7. `Dense(32, ReLU) → Dropout(0.4)`

**Tam bağlantılı (fully connected) katman:** 64 girdinin her biri 32 nöronun her birine bağlıdır.
`y = ReLU(W·x + b)`

- **ReLU** = `max(0, x)`. Negatifleri sıfırlar, doğrusal olmayan ilişkileri öğrenmeyi sağlar.
- **Görevi:** GRU'nun zaman özetini sınıflandırmaya uygun bir temsile dönüştürmek.
- **Parametre:** 64×32 + 32 = **2.080**
- Ardından yine %40 Dropout.

---

## 8. `Dense(1, Sigmoid) → P(Sepsis)`

Tek nöron + **sigmoid**: `σ(z) = 1 / (1 + e^(−z))`. Herhangi bir sayıyı **0 ile 1 arasına** sıkıştırır,
bu da **sepsis olasılığı** olarak yorumlanır. Parametre: 32 + 1 = **33**.

Bu ham skor üç aşamadan daha geçer:
1. **İzotonik kalibrasyon:** Ham %37 gerçekte %37 demek değildir (eğitimde pozitif sınıf 5 kat ağırlıklıydı).
   Doğrulama setinde öğrenilen monoton bir eşleme, skoru **gözlenen sepsis oranına** çevirir.
2. **Eşik (0.35):** Doğrulama setinde F1'i en yüksek yapan değer. Bunun üstü “sepsis riski var” demektir.
3. **Klinik güvenlik ağı:** Hipotansiyon, laktat > 2 gibi kırmızı bayraklar varsa risk sınıfı yükseltilir.

---

## Eğitim ayarları

| Ayar | Değer | Anlamı |
|------|-------|--------|
| Kayıp | Binary Cross-Entropy | İkili sınıflandırma için standart: `−[y·log p + (1−y)·log(1−p)]` |
| Sınıf ağırlığı | Pozitif × 5 | Sepsis örnekleri %1.8; ağırlık olmazsa model hep “sepsis yok” der |
| Optimizer | Adam, lr = 0.0005 | Uyarlamalı öğrenme oranlı gradyan inişi |
| clipnorm = 1 | | Gradyan normu 1'i aşarsa kırpılır; RNN'lerde patlayan gradyanı önler |
| Batch | 2048 pencere | Her güncellemede kullanılan örnek sayısı |
| Erken durdurma | val PR-AUC, sabır 3 | 3 epoch iyileşme yoksa durur, **en iyi ağırlıklara döner** |
| LR azaltma | sabır 2, × 0.5 | Takılınca öğrenme oranı yarıya iner |
| Bölme | Hasta bazlı %60 / %20 / %20 | Hiçbir test hastası eğitimde yok |

### Eğitim seyri

| Epoch | Eğitim AUC | Doğrulama AUC | Doğrulama PR-AUC |
|-------|-----------|---------------|------------------|
| 1 | 0.706 | 0.791 | 0.0963 |
| 2 | 0.777 | 0.800 | 0.0959 |
| **3** | 0.805 | **0.804** | **0.0970** ← seçilen |
| 4 | 0.829 | 0.805 | 0.0954 |
| 5 | 0.847 | 0.803 | 0.0963 |
| 6 | 0.862 | 0.783 | 0.0900 ← durdu |

3. epoch'tan sonra eğitim AUC artmaya devam ederken doğrulama düşüyor. Bu klasik **aşırı öğrenme (overfitting)**
işareti. Erken durdurma 3. epoch'un ağırlıklarını geri yükledi.

---

## Kaynaklar

- Cho K, ve ark. (2014). Learning Phrase Representations using RNN Encoder–Decoder. *EMNLP*. (GRU)
- Chung J, ve ark. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv:1412.3555*.
- Srivastava N, ve ark. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.
- Gal Y, Ghahramani Z. (2016). Dropout as a Bayesian Approximation. *ICML*. (MC-Dropout)
- Che Z, ve ark. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Scientific Reports*. (Maskeler, GRU-D)
- Reyna MA, ve ark. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/CinC Challenge 2019. *Critical Care Medicine*.
