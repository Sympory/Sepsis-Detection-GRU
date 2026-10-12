"""
Sunumdan önce sistemi uçtan uca kontrol eder.

  python sistem_kontrol.py

Kontroller: Python, paketler, model, veritabanı şeması, giriş/oturum,
rol bazlı yetkiler, denetim kaydı, tahmin (boş form, yetersiz veri,
normal ve septik hasta), alarm onayı.
Geçici bir veritabanı kullanır; gerçek patients.db'ye dokunmaz.
"""

import os
import sys
import tempfile
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['SEPSIS_DEMO_ACCOUNTS'] = '1'
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
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'HATA'}] {name}" + (f" — {detail}" if detail else ''))
    return ok


def section(title):
    print(f'\n{title}')


def main():
    print('\n=== SepsisAI sistem kontrolü ===')

    section('Ortam')
    v = sys.version_info
    check('Python sürümü', v[:2] == (3, 10), f'{v.major}.{v.minor} (3.10 gerekli: TensorFlow 2.10)')
    missing = []
    for mod in ('tensorflow', 'numpy', 'pandas', 'sklearn', 'flask', 'werkzeug'):
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

    # Geçici veritabanı
    import db
    db.DB_PATH = os.path.join(tempfile.mkdtemp(), 'kontrol.db')
    import app as sepsis_app
    import ml_service as ml
    sepsis_app.init_app()

    section('Veritabanı')
    conn = db.connect()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    need = {'users', 'sessions', 'audit_logs', 'patients', 'observations', 'measurements', 'predictions', 'alarms', 'outcomes'}
    check('Tablolar', need <= tables, f'{len(need & tables)}/{len(need)}')
    check('Demo hesapları', conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 3)
    pw = conn.execute("SELECT password_hash FROM users WHERE username='doktor'").fetchone()[0]
    check('Şifreler hash olarak saklanıyor', 'Doktor123' not in pw and len(pw) > 40, pw.split('$')[0])
    try:
        conn.execute('DELETE FROM audit_logs')
        check('Denetim kaydı silinemez', False)
    except Exception:
        check('Denetim kaydı silinemez', True)
    conn.close()

    if not check('Model yükleme', ml.load_model_and_preprocessing()):
        return finish()

    def client_as(username, password):
        c = sepsis_app.app.test_client()
        r = c.post('/api/auth/login', json={'username': username, 'password': password})
        return c, r

    section('Giriş ve oturum')
    anon = sepsis_app.app.test_client()
    check('Girişsiz ana sayfa → giriş ekranına yönlendirme', anon.get('/').status_code == 302)
    check('Girişsiz API isteği reddediliyor', anon.get('/api/patients').status_code == 401)
    check('Veritabanı dosyası tarayıcıdan erişilemez', anon.get('/patients.db').status_code == 404)
    _, r = client_as('doktor', 'yanlis-sifre')
    check('Yanlış şifre reddediliyor', r.status_code == 401, r.get_json()['error'])
    doctor, r = client_as('doktor', 'Doktor123')
    check('Doktor girişi', r.status_code == 200, r.get_json()['user']['role_label'])
    nurse, _ = client_as('hemsire', 'Hemsire123')
    admin, _ = client_as('yonetici', 'Yonetici123')
    check('Giriş sonrası ana sayfa', doctor.get('/').status_code == 200)

    section('Tahmin')
    pid = nurse.post('/api/patients', json={'patient_id': 'K1', 'name': 'Kontrol', 'age': 70, 'gender': 'Male'}).get_json()['patient_id']
    url = f'/api/patients/{pid}/hourly-data'
    check('Boş form reddediliyor', nurse.post(url, json={'hour': 1, 'vital_signs': {}}).status_code == 400)
    r = nurse.post(url, json={'hour': 1, 'vital_signs': {'HR': 90}}).get_json()
    check('Tek vital → Yetersiz Veri', r.get('risk_level') == 'Yetersiz Veri', r.get('risk_level'))
    normal = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75}
    r_norm = nurse.post(url, json={'hour': 2, 'vital_signs': normal}).get_json()
    check('Normal hasta tahmini', r_norm.get('success'),
          f"skor %{r_norm['prediction'] * 100:.1f}, {r_norm['risk_level']}, güvenilirlik {r_norm['reliability']['score']}")
    pid2 = nurse.post('/api/patients', json={'patient_id': 'K2', 'name': 'Kontrol 2', 'age': 70, 'gender': 'Male'}).get_json()['patient_id']
    septic = {'HR': 132, 'O2Sat': 88, 'Temp': 38.9, 'Resp': 29, 'SBP': 82, 'DBP': 48, 'Lactate': 4.5, 'WBC': 17}
    r_sep = nurse.post(f'/api/patients/{pid2}/hourly-data', json={'hour': 2, 'vital_signs': septic}).get_json()
    check('Septik hasta tahmini', r_sep.get('success'),
          f"skor %{r_sep['prediction'] * 100:.1f}, {r_sep['risk_level']}, güvenilirlik {r_sep['reliability']['score']}")
    check('Model septik hastaya daha yüksek skor veriyor', r_sep['prediction'] > r_norm['prediction'],
          f"%{r_norm['prediction'] * 100:.1f} → %{r_sep['prediction'] * 100:.1f}")
    check('Yüksek riskte alarm açıldı', r_sep.get('alarm_raised'))
    detail = doctor.get(f'/api/patients/{pid2}').get_json()
    check('Ölçümler normalize tablodan okunuyor', detail['hourly_data'][0]['vital_signs'].get('Lactate') == 4.5,
          f"giren: {detail['hourly_data'][0]['entered_by']}")

    section('Rol yetkileri')
    alarm_id = doctor.get('/api/alarms').get_json()['alarms'][0]['id']
    check('Hemşire alarm onaylayamaz', nurse.post(f'/api/alarms/{alarm_id}/acknowledge', json={'note': 'test'}).status_code == 403)
    check('Doktor not yazmadan onaylayamaz', doctor.post(f'/api/alarms/{alarm_id}/acknowledge', json={'note': ''}).status_code == 400)
    check('Doktor alarm onaylar', doctor.post(f'/api/alarms/{alarm_id}/acknowledge',
                                              json={'note': 'Laktat ve kültür istendi'}).status_code == 200)
    check('Yönetici veri giremez', admin.post(url, json={'hour': 3, 'vital_signs': normal}).status_code == 403)
    check('Hemşire taburcu edemez', nurse.post(f'/api/patients/{pid}/discharge').status_code == 403)
    check('Doktor kalıcı silemez', doctor.delete(f'/api/patients/{pid}').status_code == 403)
    check('Sonuç girilmeden taburcu edilemez', doctor.post(f'/api/patients/{pid}/discharge', json={}).status_code == 400)
    check('Sepsis "Evet" ise başlangıç saati zorunlu',
          doctor.post(f'/api/patients/{pid}/discharge', json={'outcome': {'sepsis': 'yes'}}).status_code == 400)
    check('Doktor taburcu eder (arşiv + sonuç)', doctor.post(f'/api/patients/{pid}/discharge',
                                                            json={'outcome': {'sepsis': 'no'}}).status_code == 200)
    check('Hemşire sonuç işaretleyemez', nurse.put(f'/api/patients/{pid}/outcome',
                                                   json={'outcome': {'sepsis': 'no'}}).status_code == 403)
    check('Taburcu hastaya veri girilemez', nurse.post(url, json={'hour': 3, 'vital_signs': normal}).status_code == 400)
    check('Hemşire denetim kayıtlarını göremez', nurse.get('/api/audit-logs').status_code == 403)
    check('Hemşire kullanıcı ekleyemez', nurse.post('/api/users', json={}).status_code == 403)
    r = admin.post('/api/users', json={'username': 'test.hemsire', 'full_name': 'Test', 'role': 'nurse', 'password': 'kisa'})
    check('Zayıf şifre reddediliyor', r.status_code == 400, r.get_json()['error'])
    check('Yönetici kullanıcı ekler', admin.post('/api/users', json={'username': 'test.hemsire', 'full_name': 'Test',
                                                                       'role': 'nurse', 'password': 'Guclu1234'}).status_code == 201)

    section('Veri seti (yeniden eğitim için)')
    doctor.post(f'/api/patients/{pid2}/discharge', json={'outcome': {'sepsis': 'yes', 'onset_hour': 8, 'note': 'kültür +'}})
    check('Doktor veri setini indiremez', doctor.get('/api/admin/dataset.csv').status_code == 403)
    r = admin.get('/api/admin/dataset.csv')
    lines = r.get_data(as_text=True).strip().splitlines()
    header = lines[0].split(',')
    check('Yönetici CSV indirir (PhysioNet formatı)', r.status_code == 200 and header[0] == 'Hour'
          and header[-2:] == ['SepsisLabel', 'Patient_ID'], f'{len(lines) - 1} satır, {len(header)} sütun')
    body = r.get_data(as_text=True)
    check('CSV kimlik bilgisi içermez', 'Kontrol' not in body and 'K1' not in body.replace('L0000', ''))
    rows = [dict(zip(header, l.split(','))) for l in lines[1:]]
    sep = [x for x in rows if x['Patient_ID'] == 'L00002']
    check('Etiket kuralı: başlangıçtan 6 saat önce SepsisLabel=1', sep and sep[0]['Hour'] == '2' and sep[0]['SepsisLabel'] == '1')
    check('Yönetici kalıcı siler', admin.delete(f'/api/patients/{pid}').status_code == 200)

    section('Denetim kaydı')
    logs = admin.get('/api/audit-logs?limit=500').get_json()
    actions = {l['action'] for l in logs['logs']}
    for act in ('auth.login', 'auth.login_failed', 'patient.create', 'observation.create', 'alarm.raised',
                'alarm.acknowledge', 'patient.discharge', 'patient.delete', 'user.create', 'auth.forbidden',
                'dataset.export'):
        check(f'Kayıt: {act}', act in actions)
    hist = doctor.get(f'/api/patients/{pid2}/history').get_json()['history']
    check('Doktor hasta geçmişini görür', len(hist) >= 2, f'{len(hist)} olay')

    r = doctor.post('/api/auth/logout')
    check('Çıkış sonrası oturum kapanır', r.status_code == 200 and doctor.get('/api/patients').status_code == 401)

    return finish()


def finish():
    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} kontrol geçti.")
    print(('TÜM KONTROLLER BAŞARILI — uygulama sunuma hazır.' if ok else 'BAZI KONTROLLER BAŞARISIZ — yukarıdaki hatalara bakın.') + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
