"""
Sunumdan önce sistemi uçtan uca kontrol eder.

  python sistem_kontrol.py

Kontroller: Python sürümü, paketler, model dosyaları, veritabanı,
gerçek bir tahmin (boş form reddi, yetersiz veri, normal ve septik hasta).
Geçici bir veritabanı kullanır; gerçek patients.db'ye dokunmaz.
"""

import os
import sys
import tempfile
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(f"  [{'OK' if ok else 'HATA'}] {name}" + (f" — {detail}" if detail else ''))
    return ok


def main():
    print('\n=== SepsisAI sistem kontrolü ===\n')

    v = sys.version_info
    check('Python sürümü', v[:2] == (3, 10), f'{v.major}.{v.minor} (3.10 gerekli: TensorFlow 2.10)')

    missing = []
    for mod in ('tensorflow', 'numpy', 'pandas', 'sklearn', 'flask', 'flask_cors'):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not check('Paketler', not missing, 'eksik: ' + ', '.join(missing) if missing else 'tamam'):
        print('\n  Çözüm:  py -3.10 -m pip install -r requirements.txt')
        return finish()

    import config
    check('Aktif model', os.path.exists(config.MODEL_PATH), f'{config.FEATURE_SET} → {os.path.relpath(config.MODEL_PATH, ROOT)}')
    check('Kalibratör', os.path.exists(os.path.join(os.path.dirname(config.MODEL_PATH), 'calibrator.pkl')))

    # Geçici veritabanı ile uygulamayı yükle
    tmp_db = os.path.join(tempfile.mkdtemp(), 'kontrol.db')
    import app as sepsis_app
    sepsis_app.DB_PATH = tmp_db
    sepsis_app.init_database()
    if not check('Model yükleme', sepsis_app.load_model_and_preprocessing()):
        return finish()

    client = sepsis_app.app.test_client()
    check('Ana sayfa', client.get('/').status_code == 200)
    check('Veritabanı dosyası tarayıcıdan erişilemez', client.get('/patients.db').status_code == 404)

    pid = client.post('/api/patients', json={'patient_id': 'K1', 'name': 'Kontrol', 'age': 70, 'gender': 'Male'}).get_json()['patient_id']
    url = f'/api/patients/{pid}/hourly-data'

    r = client.post(url, json={'hour': 1, 'vital_signs': {}})
    check('Boş form reddediliyor', r.status_code == 400)

    r = client.post(url, json={'hour': 1, 'vital_signs': {'HR': 90}}).get_json()
    check('Tek vital → Yetersiz Veri', r.get('risk_level') == 'Yetersiz Veri', r.get('risk_level'))

    normal = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75}
    r_norm = client.post(url, json={'hour': 2, 'vital_signs': normal}).get_json()
    check('Normal hasta tahmini', r_norm.get('success'),
          f"skor %{r_norm['prediction'] * 100:.1f}, {r_norm['risk_level']}, güvenilirlik {r_norm['reliability']['score']}")

    pid2 = client.post('/api/patients', json={'patient_id': 'K2', 'name': 'Kontrol 2', 'age': 70, 'gender': 'Male'}).get_json()['patient_id']
    septic = {'HR': 132, 'O2Sat': 88, 'Temp': 38.9, 'Resp': 29, 'SBP': 82, 'DBP': 48, 'Lactate': 4.5, 'WBC': 17}
    r_sep = client.post(f'/api/patients/{pid2}/hourly-data', json={'hour': 2, 'vital_signs': septic}).get_json()
    check('Septik hasta tahmini', r_sep.get('success'),
          f"skor %{r_sep['prediction'] * 100:.1f}, {r_sep['risk_level']}, güvenilirlik {r_sep['reliability']['score']}")
    check('Model septik hastaya daha yüksek skor veriyor', r_sep['prediction'] > r_norm['prediction'],
          f"%{r_norm['prediction'] * 100:.1f} → %{r_sep['prediction'] * 100:.1f}")

    info = client.get('/api/model-info').get_json()
    check('Model bilgi sayfası', info.get('success') and bool(info.get('reliability_stats')))

    return finish()


def finish():
    ok = all(results)
    print('\n' + ('TÜM KONTROLLER BAŞARILI — uygulama sunuma hazır.' if ok else 'BAZI KONTROLLER BAŞARISIZ — yukarıdaki hatalara bakın.') + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
