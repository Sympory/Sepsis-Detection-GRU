"""
Sunum için örnek hastalar oluşturur (patients.db).

  py -3.10 scripts/demo_hastalari.py          # ekle (varsa atlar)
  py -3.10 scripts/demo_hastalari.py --sil    # DEMO-* hastalarını sil

Tahminler gerçek modelle, uygulamanın kendi API'si üzerinden hesaplanır.
Uygulama açıkken de çalıştırılabilir (sonra sayfayı yenileyin).
"""

import os
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def stable_hours():
    rows = []
    for h in range(1, 9):
        v = {'HR': 76 + h % 3, 'O2Sat': 98, 'Temp': 36.7 + (h % 2) * 0.1, 'Resp': 15 + h % 2,
             'SBP': 124 - h % 3, 'DBP': 74, 'MAP': 91}
        if h == 2:
            v.update({'Lactate': 1.1, 'WBC': 7.8, 'Creatinine': 0.9, 'Platelets': 240, 'pH': 7.41})
        rows.append((h, v))
    return rows


def deteriorating_hours():
    """Ateş, taşikardi, hipotansiyon ve laktat yükselmesi ile giden tablo"""
    rows = []
    for h in range(13, 25):  # YBÜ'nün 13-24. saatleri, 12 saat boyunca yavaş kötüleşme
        i = (h - 13) * 9 / 11
        v = {'HR': 84 + i * 5, 'O2Sat': 97 - i * 1.1, 'Temp': 37.0 + i * 0.25, 'Resp': 17 + i * 1.3,
             'SBP': 122 - i * 4.5, 'DBP': 72 - i * 2.5}
        if h in (14, 19, 24):
            v.update({'Lactate': round(1.3 + i * 0.4, 1), 'WBC': round(9 + i * 1.0, 1),
                      'Creatinine': round(1.0 + i * 0.12, 2), 'Platelets': 230 - i * 12, 'pH': round(7.40 - i * 0.012, 2)})
        rows.append((h, {k: round(val, 1) if isinstance(val, float) else val for k, val in v.items()}))
    return rows


DEMO_PATIENTS = [
    ({'patient_id': 'DEMO-1', 'name': 'Ayşe Demir (stabil)', 'age': 58, 'gender': 'Female'}, stable_hours()),
    ({'patient_id': 'DEMO-2', 'name': 'Mehmet Kaya (kötüleşen)', 'age': 71, 'gender': 'Male'}, deteriorating_hours()),
    ({'patient_id': 'DEMO-3', 'name': 'Zeynep Arslan (eksik veri)', 'age': 45, 'gender': 'Female'},
     [(1, {'HR': 96, 'Temp': 37.4})]),
]


def main():
    import app as sepsis_app
    sepsis_app.init_database()
    client = sepsis_app.app.test_client()
    existing = {p['patient_id']: p['id'] for p in client.get('/api/patients').get_json()['patients']}

    if '--sil' in sys.argv:
        for info, _ in DEMO_PATIENTS:
            if info['patient_id'] in existing:
                client.delete(f"/api/patients/{existing[info['patient_id']]}")
                print(f"Silindi: {info['name']}")
        return

    if not sepsis_app.load_model_and_preprocessing():
        sys.exit(1)

    for info, hours in DEMO_PATIENTS:
        if info['patient_id'] in existing:
            print(f"Zaten var, atlandı: {info['name']}")
            continue
        pid = client.post('/api/patients', json=info).get_json()['patient_id']
        for hour, vitals in hours:
            r = client.post(f'/api/patients/{pid}/hourly-data', json={'hour': hour, 'vital_signs': vitals}).get_json()
        print(f"Eklendi: {info['name']:<30} son saat {hour}: %{r['prediction'] * 100:5.1f}  {r['risk_level']:<14} "
              f"güvenilirlik {r['reliability']['score']}")


if __name__ == '__main__':
    main()
