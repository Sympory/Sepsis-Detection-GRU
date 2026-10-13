---
title: SepsisAI
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Yoğun bakımda sepsis erken uyarı sistemi (araştırma demosu)
---

# SepsisAI — Yoğun bakımda sepsis erken uyarı (demo)

Yoğun bakım hastasının saatlik vital bulgu ve laboratuvar değerlerinden sepsis riskini tahmin eden
karar destek prototipi. Hibrit model (LightGBM + yapay sinir ağı), PhysioNet 2019 verisi (40.336 hasta),
test ROC-AUC 0.847.

**Bu bir herkese açık demodur:** gerçek hasta bilgisi girmeyin. Kayıtlar sunucu yeniden başlayınca silinir.
Araştırma prototipidir, klinik karar için kullanılmaz.

Giriş ekranındaki doktor ve hemşire demo hesaplarına tıklayarak deneyebilirsiniz. Hazır örnek hastalar:
stabil hasta, kötüleşen hasta, eksik verili hasta.

Kaynak kod: https://github.com/Sympory/Sepsis-Detection-GRU
