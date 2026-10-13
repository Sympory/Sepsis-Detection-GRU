# legacy/ — artık kullanılmayan dosyalar

Projenin önceki sürümlerinden kalan, **aktif uygulamanın kullanmadığı** dosyalar. Silinmediler; geçmişi
göstermek ve gerekirse geri dönmek için burada duruyorlar. Aktif sistem için ana dizindeki README'ye bakın.

| Klasör / dosya | Neydi | Neden kaldırıldı |
|---|---|---|
| `postgresql_auth/` (`auth.py`, `app_auth_endpoints.py`) | PostgreSQL tabanlı giriş sistemi taslağı | Hiç bağlanmamıştı; yerini `security.py` + SQLite aldı. Şema `database/schema.sql`'de duruyor |
| `run_app.py` | Eski başlatıcı | Artık var olmayan fonksiyonları çağırıyordu; yerini `baslat.bat` / `app.py` aldı |
| `train_gru_v23.py`, `run_gru_on_csv_v23.py`, `example_usage.py` | İlk GRU modeli (v23) | Veri saat bazında rastgele bölünmüştü (sızıntı, iyimser 0.88) |
| `evaluation_results_v23/` | v23 grafikleri | Aynı sızıntılı değerlendirme |
| `scripts/generate_biomarker_*.py`, `prepare_dataset_phase3.py`, `train_phase3.py` | Sentetik biyobelirteç (PCT, CRP…) üretimi | Etiketle ilişkili sentetik veri → 0.9997 AUC (sızıntı) |
| `ui/`, `insert_biomarkers.py`, `archive` | Eski arayüz parçaları | Arayüz yeniden yazıldı |
| `docker/` | Docker, docker-compose, nginx | `run_app.py`'yi ve kullanılmayan PostgreSQL'i çağırıyordu; Docker yol haritasında |
| `docs/` | Eski kurulum ve güvenlik notları | Eski sisteme aitti |
| `requirements_auth.txt`, `requirements_full.txt` | Eski bağımlılık listeleri | Tek `requirements.txt` kullanılıyor |

Eski belgeler (0.88 iddiasını içerenler) `docs/eski/` altında.
