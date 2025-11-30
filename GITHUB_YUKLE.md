# 🚀 GitHub'a Yükleme Talimatları

## Projeniz GitHub'a Hazır! ✅

Aşağıdaki adımları takip ederek projenizi GitHub'a yükleyebilirsiniz.

---

## Adım 1: GitHub'da Repository Oluşturun

1. **GitHub'a gidin**: https://github.com
2. Sağ üst köşedeki **"+"** işaretine tıklayın
3. **"New repository"** seçin
4. Repository bilgilerini girin:
   - **Repository name**: `sepsis-detection-gru` (veya istediğiniz isim)
   - **Description**: `GRU-based deep learning system for early sepsis detection in ICU patients`
   - **Public** veya **Private** seçin
   - ⚠️ **ÖNEMLI**: "Initialize with README", ".gitignore", veya "license" seçeneklerini **SEÇMEYİN** (zaten var)
5. **"Create repository"** butonuna tıklayın

---

## Adım 2: Local Repository'nizi GitHub'a Bağlayın

GitHub'da oluşturduğunuz repository sayfasında gösterilen komutları kullanın:

### Windows CMD/PowerShell:

```bash
# Projenizin bulunduğu klasöre gidin

# GitHub repository'nizi remote olarak ekleyin
# ⚠️ KULLANICI_ADINIZ ve REPO_ADINIZ yerine kendi bilgilerinizi yazın
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git

# Ana dalı 'main' olarak ayarlayın (GitHub standartı)
git branch -M main

# İlk push'u yapın
git push -u origin main
```

### Örnek:
```bash
git remote add origin https://github.com/ahmet/sepsis-detection-gru.git
git branch -M main
git push -u origin main
```

GitHub şifrenizi soracaktır. Eğer 2FA (iki faktörlü doğrulama) kullanıyorsanız, **Personal Access Token** oluşturmanız gerekebilir.

---

## Adım 3: Personal Access Token (Gerekiyorsa)

GitHub artık şifre ile push işlemini desteklemiyor. Token oluşturmak için:

1. GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. **"Generate new token"** tıklayın
3. **Scopes**: `repo` seçeneğini işaretleyin
4. Token'ı kopyalayın (bu ekran bir daha gösterilmeyecek!)
5. Push yaparken şifre yerine bu token'ı kullanın

---

## Adım 4: Doğrulama

Push işlemi tamamlandıktan sonra:

1. GitHub repository sayfanızı yenileyin
2. Tüm dosyalarınızın yüklendiğini kontrol edin
3. README.md'nin güzel görüntülendiğinden emin olun

---

## 🎉 Tebrikler!

Projeniz artık GitHub'da! Şimdi yapabilecekleriniz:

### ✅ Sonraki Adımlar

1. **Repository ayarları**:
   - "About" bölümünü düzenleyin (Description, Website, Topics)
   - Topics ekleyin: `deep-learning`, `healthcare`, `sepsis`, `tensorflow`, `gru`, `machine-learning`, `icu`

2. **README.md'yi düzenleyin**:
   - `yourusername` yerine gerçek kullanıcı adınızı yazın
   - Repository URL'lerini güncelleyin
   - İletişim bilgilerinizi ekleyin

3. **GitHub Pages** (Opsiyonel):
   - Settings → Pages → Source: `main` branch seçin
   - Dokümantasyonunuz için web sitesi oluşturabilirsiniz

4. **Releases** oluşturun:
   - Releases → "Create a new release"
   - Tag: `v1.0.0`
   - Model dosyasını release'e ekleyebilirsiniz

---

## 📋 Gelecek Güncellemeler İçin

Proje üzerinde değişiklik yaptığınızda:

```bash
# Değişiklikleri staging'e ekle
git add .

# Commit mesajı ile kaydet
git commit -m "Update: Yaptığınız değişikliğin açıklaması"

# GitHub'a push et
git push origin main
```

**Commit mesaj önerileri:**
- `Add: Yeni özellik eklendi`
- `Fix: Bug düzeltildi`
- `Update: Mevcut özellik güncellendi`
- `Docs: Dokümantasyon değişikliği`
- `Refactor: Kod iyileştirmesi`

---

## 🐛 Sorun Giderme

### Problem: "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
```

### Problem: "Authentication failed"
- Personal Access Token kullanın (yukarıya bakın)
- Token'ın `repo` yetkisine sahip olduğundan emin olun

### Problem: "Repository not found"
- URL'nin doğru olduğundan emin olun
- Repository'nin public/private ayarlarını kontrol edin

---

## 📊 Projenizin İçeriği

✅ 24 dosya commit edildi:
- Python script'leri (preprocessing, training, inference, web app)
- Web arayüzü (HTML, CSS, JavaScript)
- Dokümantasyon (README, LICENSE, CONTRIBUTING)
- Model dosyaları ve sonuçları
- Git konfigürasyonu (.gitignore)

❌ Commit edilmeyen dosyalar (.gitignore tarafından):
- .venv/ (virtual environment)
- *.db (SQLite veritabanları)
- Büyük model dosyaları (*.keras files - GitHub LFS gerektirebilir)
- Büyük veri dosyaları (*.csv files)

> **Not**: Model dosyasını (gru_v23_best.keras) GitHub'a yüklemek isterseniz, GitHub LFS (Large File Storage) kullanmalısınız veya GitHub Releases kullanabilirsiniz.

---

## 🎓 Ekstra: GitHub LFS (Büyük Dosyalar İçin)

Model dosyanız 100MB'dan büyükse:

```bash
# Git LFS'i kurun (Windows için: https://git-lfs.github.com/)

# LFS'i başlatın
git lfs install

# Model dosyalarını track edin
git lfs track "*.keras"
git lfs track "*.h5"

# .gitattributes dosyasını commit edin
git add .gitattributes
git commit -m "Add: Git LFS tracking for model files"

# Model dosyasını ekleyin
git add models/gru_v23_best.keras
git commit -m "Add: Trained GRU model"
git push origin main
```

---

## 📧 İletişim

Sorularınız için:
- GitHub Issues: Repository sayfanızda "Issues" sekmesi
- E-posta: README.md'de belirtin

---

**Hazırlayan**: Gemini AI  
**Tarih**: 30 Kasım 2025  
**Proje**: GRU-Based Sepsis Detection System
