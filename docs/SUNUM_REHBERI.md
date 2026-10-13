# Sunum Rehberi — SepsisAI

## Sunumdan önce (5 dakika)

1. **`sistem_kontrol.bat`** → sonunda “TÜM KONTROLLER BAŞARILI” yazmalı.
2. **`baslat.bat`** → ~20 sn sonra tarayıcıda giriş ekranı açılır. Siyah pencereyi **kapatmayın**, sunucu odur.
   Giriş ekranındaki demo hesaplarına tıklayınca bilgiler dolar: `hemsire / Hemsire123`, `doktor / Doktor123`, `yonetici / Yonetici123`.
3. Demo hastalar listede yoksa: `py -3.10 scripts/demo_hastalari.py`, sonra sayfayı yenileyin.
4. Projektör karanlıksa sol alttaki **Tema** düğmesiyle koyu moda geçin.

> Sorun çıkarsa: siyah pencereyi kapatıp `baslat.bat`'ı tekrar açın. “5000 portu kullanımda” yazıyorsa
> önceki pencere hâlâ açıktır.

---

## Konuşma akışı (~10 dakika)

### 1. Problem (1 dk)
- Sepsis: enfeksiyona karşı vücudun kontrolsüz yanıtı. Yoğun bakımda önde gelen ölüm nedenlerinden biri.
- Her saat gecikme mortaliteyi artırır → **erken uyarı** değerli.
- Soru: Saatlik vital ve lab verilerinden sepsisi önceden tahmin edebilir miyiz?

### 2. Veri (1 dk)
- PhysioNet / Computing in Cardiology Challenge 2019 (Reyna ve ark., *Critical Care Medicine* 2020).
- **40.336 hasta, 1,55 milyon saatlik kayıt**, 34 klinik değişken + yaş, cinsiyet.
- Zorluklar: Sepsis oranı sadece **%1.8**. Laboratuvar değerlerinin **%90'dan fazlası eksik**.

### 3. Model (1,5 dk)
- **Başlangıç:** GRU (zaman serisi için tekrarlayan sinir ağı), son 6 saatin ham değerleri → ROC-AUC 0.806.
- **Şimdi: hibrit model.** Hastanın tüm geçmişinden **281 zaman serisi özelliği** (24 saatlik eğilim, son ölçümden
  geçen süre, SIRS/SOFA, şok indeksi) → **sinir ağı (MLP)** + **LightGBM**, ağırlıklı ortalama → **0.847**.
- Sinir ağı: Dense 256 → 128 → 64 → Sigmoid; BatchNorm, Dropout 0.5, L2, girdi gürültüsü. ~115 bin ağırlık.
- LightGBM ağaçları uygulamada saf NumPy ile çalışıyor (Windows'ta TensorFlow ile DLL çakışması yaşanmasın diye).

### 4. Canlı demo (3 dk)
0. **Roller** (hocanın istediği): önce **hemşire** olarak girin → veri girin; alarm onaylayamadığını gösterin.
   Çıkış yapıp **doktor** olarak girin → Alarmlar'da not yazıp onaylayın, bir hastayı taburcu edin (arşive düşer).
   **Yönetici** olarak girin → Kullanıcılar (rol tablosu), Denetim kayıtları (kim ne yaptı), Veritabanı (tablolar ve ilişkiler).
1. **Genel Bakış**: hastalar riske göre sıralı; güvenilirlik sütunu.
2. **Mehmet Kaya (kötüleşen)**: risk trendi 12 saatte %2'den %42'ye çıkıyor, 22. saatte eşiği geçiyor. Gölgeli alan belirsizlik;
   panelde LightGBM ve sinir ağı dallarının skorları ayrı görünüyor.
   Sağ panel: skor, güvenilirlik halkası, “Neden?” listesi.
3. **Zeynep Arslan (eksik veri)**: sadece 2 vital → **“Değerlendirilemedi”**.
   *“İlk sürüm burada ‘hasta değil’ diyordu; boş alanları ortalama değerle doldurduğu için.”*
4. **Canlı giriş — Ayşe Demir (stabil)**. Form saati otomatik 9 yapar.
   - Sadece **nabız 110** → “Kaydet” → **Yetersiz Veri**. *“Eksik veriyle karar vermiyor.”*
   - Aynı saat için **nabız 125, ateş 39.2, SpO2 89, solunum 28** (üzerine yaz: Evet) →
     model skoru yalnızca **~%3.5**, ama risk **Yüksek**: klinik kurallar devreye girdi ve panel
     “Model bu tabloyu yakalayamadı” diyor. *“Model son 6 saatin eğilimine bakar; önceki 8 saat
     normaldi. Tek kötü saat için klinik güvenlik ağı var.”*
   - Saat 10: **nabız 128, ateş 39.3, SpO2 88, solunum 29, SKB 86, DKB 50** → model skoru **~%23**'e
     çıkar. *“Kötüleşme sürdükçe model de yakalıyor.”*
   - Sunumdan sonra temizlemek için: hastayı silip `py -3.10 scripts/demo_hastalari.py` ile yeniden oluşturun.
5. **Model & Güvenilirlik** sayfası: karşılaştırma tablosu ve klinik tutarlılık testi.

> Risk trendi grafiği hasta sayfasının en üstünde, hasta adının hemen altında.

### 5. Öğrenilenler (1 dk) — **hocaların en çok değer verdiği kısım**
- İlk model **0.88 AUC** gösteriyordu ama veri **rastgele** bölünmüştü. Aynı hasta hem eğitimde hem testteydi.
- Hasta bazlı bölmede gerçek değer **~0.81**.
- Eski model septik hastaya %7 veriyordu: “değer ne?” yerine “ölçülmüş mü?” sorusunu öğrenmişti.
- Çözüm 1: ileri doldurma + ölçüm maskeleri → septik tabloya artık yüksek risk.
- Çözüm 2: eğilimleri açıkça hesaplayan özellikler + sinir ağı/LightGBM birleşimi → 0.806 → **0.847**.
  Kazancın çoğu model türünden değil **özellik mühendisliğinden** geldi.
- **Ders:** Yüksek metrik ≠ iyi model. Değerlendirme yöntemi metrik kadar önemli.

### 6. Önümüzdeki 6 hafta (1,5 dk) — *“bitmiş proje değil, sağlam bir başlangıç”*
- **Bulgu:** 7 farklı mimari hasta bazlı testte 0.80–0.81'de toplanıyor. Sorun mimaride değil, özelliklerde ve değerlendirmede.
- **Plan** (`docs/PROJE_PLANI_6_HAFTA.md`):
  1. ✅ Değerlendirme: 5 katlı CV (0.842 ± 0.009), Utility 0.427 (NEWS2 0.115, SIRS 0.054, qSOFA 0.023)
  2. Özellikler: ölçümden geçen süre, 12/24 saatlik trendler, SOFA
  3. Mimari: uzun pencere, GRU-D, TCN, Transformer
  4. Ensemble (GRU + LightGBM), kayıp fonksiyonları
  5. ◐ Açıklanabilirlik: tahmin başına faktör açıklaması yapıldı; SHAP, conformal prediction kaldı
  6. Dış doğrulama (hastane A → B), rapor
- **Hedef:** AUC 0.83–0.85, Utility ≥ 0.30 (yarışma kazananı 0.36).
- Model GPU'da 3 dakikada eğitildiği için 6 haftada yüzlerce deney yapılabilir.
- Araştırma prototipi, klinik kullanım için değil.

---

## Olası sorular

**Mimarideki sayılar ne demek? (64, 128, 0.4, 1e-4…)**
`docs/MIMARI_ACIKLAMA.md` dosyasında satır satır var. Kısaca: 64 = GRU'nun hafıza boyutu, 128 = iki yönün
birleşimi (64+64), 0.4 = nöronların %40'ını rastgele kapatma, 1e-4 = L2 ceza katsayısı, 0.1 = girdi gürültüsünün
standart sapması, 32 = ara katman nöron sayısı, Sigmoid = çıktıyı 0–1 olasılığına çevirir.

**Neden 6 saatlik pencere?**
Veri setinde sepsis etiketi, klinik tanıdan 6 saat önce başlıyor; amaç 6 saat önceden uyarmak. Daha uzun
pencereler (12–48 saat) 3. haftada denenecek.

**Model yeniden eğitilebilir mi?**
Evet, tek komut: `py -3.10 scripts/train_v3.py`. GPU'da ~3 dakika. Metrikler, kalibrasyon ve testler otomatik çıkıyor.

**56 özellikli model neden 0.9997 aldı?**
Veri setine eklenmiş PCT, CRP, Presepsin gibi biyobelirteçler sentetikti ve etiketle ilişkiliydi, yani sızıntı.
Gerçek hayatta bu kadar mükemmel bir model olmaz; bu yüzden çıkarıldılar.

**Neden GRU, LSTM değil?**
Daha az parametre (2 kapı ve 3 kapı), daha hızlı eğitim, bu boyutta veride benzer performans.

**AUC 0.81 düşük değil mi?**
PhysioNet 2019 yarışmasında saatlik tahmin zor bir problem. Hasta bazlı dürüst değerlendirmede bu bant
literatürle uyumlu. Aynı veride LightGBM 0.811 aldı; model veriyle sınırlı.

**PR-AUC neden 0.132?**
Pozitif oran %1.8. Rastgele tahminin PR-AUC'si 0.018, model bunun ~7 katı. Dengesiz veride
PR-AUC, ROC-AUC'den daha zorlayıcıdır.

**Duyarlılık neden %31?**
Varsayılan eşik (0.33) yanlış alarmı az tutmak için seçildi. Tarama eşiğinde (0.066) duyarlılık %81 olur,
özgüllük %75'e düşer (ilk modelde %64'tü). Bu bir klinik tercih; iki seçenek de model sayfasında gösteriliyor.

**Güvenilirlik skoru nasıl hesaplanıyor?**
Dört bileşen: veri tamlığı (%45), 6 saatlik pencerenin kapsamı (%20), MC-Dropout ile model kararlılığı (%25),
eğitim dağılımına uyum (%10). Ayrıca en az 3 temel vital yoksa tahmin hiç verilmiyor.

**MC-Dropout nedir?**
Tahmin sırasında Dropout'u açık bırakıp modeli 30 kez çalıştırıyoruz. Sonuçlar birbirine yakınsa model
emin, dağınıksa kararsız (Gal ve Ghahramani, 2016). Testte kararlı tahminlerin doğruluğu %61, kararsızların %45.

**Kalibrasyon ne işe yarıyor?**
Ham skor %42 olsa bile bu, hastaların %42'sinin sepsis olduğu anlamına gelmez. İzotonik regresyon skoru
gerçek gözlenen orana çeviriyor (hata 0.040'tan 0.001'e düştü).

**Python'da neler kullandın?**
TensorFlow/Keras (model), pandas/NumPy (veri), scikit-learn (izotonik kalibrasyon, metrikler),
Flask (REST API), SQLite (veritabanı), matplotlib (grafikler). Ön yüz HTML/CSS/JavaScript ve Chart.js.

**Veri sızıntısını nasıl fark ettin?**
Eski 0.88'lik model, eğitimde görmüş olduğu hastalardan rastgele seçilen 4.000 hastada 0.94 aldı. Aynı mimariyle
hasta bazlı bölmede eğitilen modeller ise hep ~0.80–0.81'de kaldı. Ayrıca yeni modelde eğitim AUC 0.92'ye çıkarken validasyon 0.75'e düşünce statik özelliklerin hastayı
tanımladığını buldum.
