"""
Sepsis Tahmin Sistemi — Web Sunucusu (Flask)
============================================

Katmanlar:
    app.py         → HTTP rotaları (bu dosya)
    security.py    → giriş, oturum, rol bazlı yetki
    db.py          → SQLite şeması, taşıma, sorgular, denetim kaydı
    ml_service.py  → model, tahmin, güvenilirlik

Roller: Yönetici (admin), Doktor (doctor), Hemşire (nurse). İzinler security.PERMISSIONS'ta.
"""

import json
import os
import sqlite3
import traceback

from flask import Flask, g, jsonify, redirect, request, send_from_directory
from werkzeug.exceptions import HTTPException

import db
import ml_service as ml
import security
from config import FEATURE_SET, HOST, MODEL_INFO, OPTIMAL_THRESHOLD, PORT
from features import feature_names
from reliability import normalize_inputs
from security import require

app = Flask(__name__, static_folder=None)
MODEL_VERSION = f"{FEATURE_SET} · {MODEL_INFO.get('version', '')}"

# Yalnızca arayüz dosyaları servis edilir; proje klasöründeki diğer dosyalar
# (patients.db, .env, modeller) dışarı açılmaz.
STATIC_FILES = {'style.css', 'script.js', 'login.js'}


@app.before_request
def open_db():
    g.db = db.connect()
    g.user = None


@app.teardown_request
def close_db(exc):
    conn = g.pop('db', None)
    if conn is not None:
        conn.close()


@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return fail(e.description or e.name, e.code)
    traceback.print_exc()
    return jsonify({'success': False, 'error': 'Sunucu hatası: ' + str(e)}), 500


def ok(payload=None, status=200):
    return jsonify({'success': True, **(payload or {})}), status


def fail(message, status=400, **extra):
    return jsonify({'success': False, 'error': message, **extra}), status


def audit(action, **kwargs):
    db.audit(g.db, g.user, action, ip=security.client_ip(), **kwargs)


# ============================================================================
# SAYFALAR
# ============================================================================

@app.route('/')
def index():
    if not security.load_user(g.db):
        return redirect('/login')
    return send_from_directory('.', 'index.html')


@app.route('/login')
def login_page():
    if security.load_user(g.db):
        return redirect('/')
    return send_from_directory('.', 'login.html')


@app.route('/<path:path>')
def serve_static(path):
    if path not in STATIC_FILES:
        return fail('Bulunamadı', 404)
    return send_from_directory('.', path)


# ============================================================================
# KİMLİK DOĞRULAMA
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        return fail('Kullanıcı adı ve şifre gerekli')

    user, error = security.authenticate(g.db, username, password)
    if not user:
        db.audit(g.db, None, 'auth.login_failed', ip=security.client_ip(), success=False,
                 details={'username': username[:50], 'reason': error})
        g.db.commit()
        return fail(error, 401)

    token = security.create_session(g.db, user)
    g.user = dict(user)
    audit('auth.login', resource_type='user', resource_id=user['id'])
    g.db.commit()
    response, status = ok({'user': security.public_user(user)})
    return security.set_session_cookie(response, token), status


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    user = security.load_user(g.db)
    security.end_session(g.db, request.cookies.get(security.SESSION_COOKIE))
    if user:
        g.user = user
        audit('auth.logout', resource_type='user', resource_id=user['id'])
    g.db.commit()
    response, status = ok()
    response.delete_cookie(security.SESSION_COOKIE)
    return response, status


@app.route('/api/auth/me')
def me():
    user = security.load_user(g.db)
    if not user:
        return fail('Oturum açmanız gerekiyor', 401, code='auth_required')
    return ok({'user': security.public_user(user), 'session_idle_minutes': security.SESSION_IDLE_MINUTES,
               'public_demo': security.public_demo()})


@app.route('/api/auth/change-password', methods=['POST'])
@require('patients.view')
def change_password():
    data = request.get_json(silent=True) or {}
    if not security.verify_password(data.get('current_password') or '', g.user['password_hash']):
        audit('auth.password_change', resource_type='user', resource_id=g.user['id'], success=False)
        g.db.commit()
        return fail('Mevcut şifre hatalı')
    problem = security.password_problem(data.get('new_password'))
    if problem:
        return fail(problem)
    g.db.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                 (security.hash_password(data['new_password']), g.user['id']))
    audit('auth.password_change', resource_type='user', resource_id=g.user['id'])
    g.db.commit()
    return ok({'message': 'Şifre değiştirildi'})


@app.route('/api/auth/demo-accounts')
def demo_accounts():
    return ok({'public_demo': security.public_demo(),
               'accounts': [{'username': a['username'], 'password': a['password'],
                             'role_label': security.ROLE_LABELS[a['role']]} for a in security.visible_demo_accounts()]})


# ============================================================================
# HASTALAR
# ============================================================================

@app.route('/api/patients', methods=['GET'])
@require('patients.view')
def list_patients():
    status = request.args.get('status', 'active')
    if status not in ('active', 'discharged', 'all'):
        return fail('Geçersiz durum filtresi')
    return ok({'patients': db.list_patients(g.db, status)})


@app.route('/api/patients', methods=['POST'])
@require('patients.create')
def create_patient():
    data = request.get_json(silent=True) or {}
    code = (data.get('patient_id') or '').strip()
    name = (data.get('name') or '').strip()
    if not code or not name:
        return fail('Hasta ID ve ad zorunlu')
    age = data.get('age')
    if age not in (None, ''):
        try:
            age = int(age)
        except (TypeError, ValueError):
            return fail('Yaş sayı olmalı')
        if not 0 <= age <= 120:
            return fail('Yaş 0–120 arasında olmalı')
    else:
        age = None
    gender = data.get('gender') or None
    if gender not in (None, 'Male', 'Female'):
        return fail('Cinsiyet geçersiz')

    try:
        pk = g.db.execute(
            'INSERT INTO patients (patient_id, name, age, gender, admission_time, created_at, created_by) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (code, name, age, gender, data.get('admission_time') or db.now(), db.now(), g.user['id'])).lastrowid
    except sqlite3.IntegrityError:
        return fail('Bu hasta ID zaten kayıtlı')
    audit('patient.create', resource_type='patient', resource_id=pk, patient_id=pk,
          details={'patient_id': code, 'name': name, 'age': age, 'gender': gender})
    g.db.commit()
    return ok({'patient_id': pk, 'message': 'Hasta kabul edildi'}, 201)


@app.route('/api/patients/<int:patient_pk>', methods=['GET'])
@require('patients.view')
def get_patient(patient_pk):
    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    open_alarm = g.db.execute("SELECT * FROM alarms WHERE patient_id = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
                              (patient_pk,)).fetchone()
    return ok({'patient': patient, 'hourly_data': db.get_observations(g.db, patient_pk),
               'open_alarm': dict(open_alarm) if open_alarm else None})


@app.route('/api/patients/<int:patient_pk>/hourly-data', methods=['POST'])
@require('observations.create')
def add_hourly_data(patient_pk):
    """Saatlik ölçüm ekler (veya aynı saati günceller), tahmin + güvenilirlik hesaplar."""
    if not ml.is_ready():
        return fail('Model yüklenmedi', 503)
    data = request.get_json(silent=True) or {}
    if 'hour' not in data or 'vital_signs' not in data or not isinstance(data['vital_signs'], dict):
        return fail('hour ve vital_signs gerekli')
    try:
        hour = int(data['hour'])
    except (TypeError, ValueError):
        return fail('Saat geçerli bir tam sayı olmalı')
    if hour < 0:
        return fail('Saat negatif olamaz')
    if hour > ml.MAX_HOUR:
        return fail(f'Saat en fazla {ml.MAX_HOUR} olabilir')

    is_valid, errors = ml.validate_vital_signs(data['vital_signs'])
    if not is_valid:
        return fail('Geçersiz değerler', details=errors)
    vital_signs, notes = normalize_inputs(data['vital_signs'])
    if not any(k not in ml.MODEL_UNUSED_FIELDS for k in vital_signs):
        return fail('Hiç ölçüm girilmedi. Boş form modele gönderilirse eksikler "normal" '
                    'medyan değerlerle doldurulur ve sonuç yanıltıcı olur.')

    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    if patient['status'] != 'active':
        return fail('Taburcu edilmiş hastaya veri girilemez')

    demographics = {}
    if patient['gender']:
        demographics['Gender'] = 1 if patient['gender'] == 'Male' else 0
    if patient['age'] is not None:
        demographics['Age'] = patient['age']
    previous = [(o['hour'], o['vital_signs']) for o in db.get_observations(g.db, patient_pk, before_hour=hour)]

    result = ml.assess(previous, hour, vital_signs, demographics, notes)
    obs_id, old_values = db.save_observation(g.db, patient_pk, hour, vital_signs, g.user, result, MODEL_VERSION)
    alarm_id = db.raise_alarm_if_needed(g.db, patient_pk, obs_id, result)

    audit('observation.update' if old_values is not None else 'observation.create',
          resource_type='observation', resource_id=obs_id, patient_id=patient_pk,
          details={'hour': hour, 'values': vital_signs, 'old_values': old_values,
                   'risk_level': result['risk_level'], 'score': round(result['prediction'], 4),
                   'reliability': result['assessment']['score']})
    if alarm_id:
        db.audit(g.db, None, 'alarm.raised', resource_type='alarm', resource_id=alarm_id, patient_id=patient_pk,
                 details={'risk_level': result['risk_level'], 'hour': hour})
    g.db.commit()

    return ok({'hour': hour, 'prediction': result['prediction'], 'risk_level': result['risk_level'],
               'risk_color': result['risk_color'], 'is_sepsis_risk': result['is_sepsis_risk'],
               'threshold': OPTIMAL_THRESHOLD, 'reliability': result['assessment'],
               'alarm_raised': bool(alarm_id), 'overwritten': old_values is not None,
               'message': 'Veri kaydedildi'}, 201)


def _parse_outcome(data, patient_pk):
    """Doktorun girdiği sonucu doğrular. Dönüş: ((sepsis, onset_hour, note), hata)"""
    outcome = data.get('outcome') or {}
    sepsis = outcome.get('sepsis')
    if sepsis not in ('yes', 'no', 'unknown'):
        return None, 'Sepsis sonucu seçilmeli (Evet / Hayır / Belirsiz)'
    onset = outcome.get('onset_hour')
    if sepsis == 'yes' and outcome.get('onset_unknown'):
        onset = None                                   # sepsis kesin, saat bilinmiyor → veri setine alınmaz
    elif sepsis == 'yes':
        try:
            onset = int(onset)
        except (TypeError, ValueError):
            return None, 'Sepsis başlangıç saati (YBÜ saati) girilmeli'
        if onset < 0:
            return None, 'Başlangıç saati negatif olamaz'
        last = g.db.execute('SELECT MAX(hour) FROM observations WHERE patient_id = ?', (patient_pk,)).fetchone()[0]
        if last is not None and onset > last + 48:
            return None, f'Başlangıç saati son kayıttan (saat {last}) çok sonra olamaz'
    else:
        onset = None
    note = (outcome.get('note') or '').strip()[:500] or None
    return (sepsis, onset, note), None


@app.route('/api/patients/<int:patient_pk>/discharge', methods=['POST'])
@require('patients.discharge')
def discharge_patient(patient_pk):
    """Taburcu: hasta arşive alınır, doktor gerçek sonucu (veri seti etiketi) işaretler."""
    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    if patient['status'] == 'discharged':
        return fail('Hasta zaten taburcu edilmiş')
    data = request.get_json(silent=True) or {}
    outcome, error = _parse_outcome(data, patient_pk)
    if error:
        return fail(error)
    g.db.execute("UPDATE patients SET status = 'discharged', discharged_at = ?, discharged_by = ? WHERE id = ?",
                 (db.now(), g.user['id'], patient_pk))
    db.save_outcome(g.db, patient_pk, *outcome, g.user)
    closed = g.db.execute(
        "UPDATE alarms SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = ?, "
        "note = 'Hasta taburcu edildi' WHERE patient_id = ? AND status = 'open'",
        (g.user['id'], db.now(), patient_pk)).rowcount
    audit('patient.discharge', resource_type='patient', resource_id=patient_pk, patient_id=patient_pk,
          details={'sepsis': outcome[0], 'onset_hour': outcome[1], 'note': outcome[2], 'closed_alarms': closed})
    g.db.commit()
    return ok({'message': 'Hasta taburcu edildi ve arşive alındı'})


@app.route('/api/patients/<int:patient_pk>/outcome', methods=['PUT'])
@require('outcomes.label')
def update_outcome(patient_pk):
    """Taburcu edilmiş hastanın sonucunu düzeltir (ör. kültür sonucu sonradan geldiyse)."""
    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    outcome, error = _parse_outcome(request.get_json(silent=True) or {}, patient_pk)
    if error:
        return fail(error)
    old = db.save_outcome(g.db, patient_pk, *outcome, g.user)
    audit('patient.outcome', resource_type='patient', resource_id=patient_pk, patient_id=patient_pk,
          details={'sepsis': outcome[0], 'onset_hour': outcome[1], 'note': outcome[2], 'old': old})
    g.db.commit()
    return ok({'message': 'Sonuç güncellendi'})


@app.route('/api/patients/<int:patient_pk>/readmit', methods=['POST'])
@require('patients.discharge')
def readmit_patient(patient_pk):
    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    if patient['status'] == 'active':
        return fail('Hasta zaten aktif')
    g.db.execute("UPDATE patients SET status = 'active', discharged_at = NULL, discharged_by = NULL WHERE id = ?",
                 (patient_pk,))
    audit('patient.readmit', resource_type='patient', resource_id=patient_pk, patient_id=patient_pk)
    g.db.commit()
    return ok({'message': 'Hasta yeniden aktif'})


@app.route('/api/patients/<int:patient_pk>', methods=['DELETE'])
@require('patients.delete')
def delete_patient(patient_pk):
    patient = db.get_patient(g.db, patient_pk)
    if not patient:
        return fail('Hasta bulunamadı', 404)
    hours = g.db.execute('SELECT COUNT(*) FROM observations WHERE patient_id = ?', (patient_pk,)).fetchone()[0]
    g.db.execute('DELETE FROM patients WHERE id = ?', (patient_pk,))   # ölçümler, tahminler, alarmlar CASCADE
    audit('patient.delete', resource_type='patient', resource_id=patient_pk, patient_id=patient_pk,
          details={'patient_id': patient['patient_id'], 'name': patient['name'], 'deleted_hours': hours})
    g.db.commit()
    return ok({'message': 'Hasta ve tüm kayıtları kalıcı olarak silindi'})


@app.route('/api/patients/<int:patient_pk>/history')
@require('history.view')
def patient_history(patient_pk):
    return ok({'history': db.patient_history(g.db, patient_pk)})


# ============================================================================
# ALARMLAR
# ============================================================================

@app.route('/api/alarms')
@require('alarms.view')
def list_alarms():
    status = request.args.get('status', 'open')
    if status not in ('open', 'acknowledged', 'all'):
        return fail('Geçersiz durum filtresi')
    return ok({'alarms': db.list_alarms(g.db, status)})


@app.route('/api/alarms/<int:alarm_id>/acknowledge', methods=['POST'])
@require('alarms.acknowledge')
def acknowledge_alarm(alarm_id):
    alarm = g.db.execute('SELECT * FROM alarms WHERE id = ?', (alarm_id,)).fetchone()
    if not alarm:
        return fail('Alarm bulunamadı', 404)
    if alarm['status'] != 'open':
        return fail('Alarm zaten onaylanmış')
    note = ((request.get_json(silent=True) or {}).get('note') or '').strip()
    if len(note) < 3:
        return fail('Klinik not zorunlu (ör. "Laktat ve kültür istendi")')
    g.db.execute("UPDATE alarms SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = ?, note = ? WHERE id = ?",
                 (g.user['id'], db.now(), note[:1000], alarm_id))
    audit('alarm.acknowledge', resource_type='alarm', resource_id=alarm_id, patient_id=alarm['patient_id'],
          details={'note': note[:1000], 'risk_level': alarm['risk_level']})
    g.db.commit()
    return ok({'message': 'Alarm onaylandı'})


# ============================================================================
# YÖNETİCİ: KULLANICILAR VE DENETİM KAYITLARI
# ============================================================================

def _user_row(u):
    return {'id': u['id'], 'username': u['username'], 'full_name': u['full_name'], 'role': u['role'],
            'role_label': security.ROLE_LABELS.get(u['role']), 'is_active': bool(u['is_active']),
            'locked_until': u['locked_until'], 'last_login': u['last_login'], 'created_at': u['created_at']}


@app.route('/api/users')
@require('users.manage')
def list_users():
    rows = g.db.execute('SELECT * FROM users ORDER BY role, full_name').fetchall()
    return ok({'users': [_user_row(u) for u in rows]})


@app.route('/api/users', methods=['POST'])
@require('users.manage')
def create_user():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    full_name = (data.get('full_name') or '').strip()
    role = data.get('role')
    if not username or not full_name:
        return fail('Kullanıcı adı ve ad soyad zorunlu')
    if role not in security.ROLE_LABELS:
        return fail('Geçersiz rol')
    problem = security.password_problem(data.get('password'))
    if problem:
        return fail(problem)
    try:
        uid = g.db.execute('INSERT INTO users (username, password_hash, full_name, role, created_by) VALUES (?, ?, ?, ?, ?)',
                           (username, security.hash_password(data['password']), full_name, role, g.user['id'])).lastrowid
    except sqlite3.IntegrityError:
        return fail('Bu kullanıcı adı zaten var')
    audit('user.create', resource_type='user', resource_id=uid,
          details={'username': username, 'full_name': full_name, 'role': role})
    g.db.commit()
    return ok({'user_id': uid}, 201)


@app.route('/api/users/<int:user_id>', methods=['PATCH'])
@require('users.manage')
def update_user(user_id):
    target = g.db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not target:
        return fail('Kullanıcı bulunamadı', 404)
    data = request.get_json(silent=True) or {}
    changes = {}
    if 'role' in data and data['role'] != target['role']:
        if data['role'] not in security.ROLE_LABELS:
            return fail('Geçersiz rol')
        if user_id == g.user['id']:
            return fail('Kendi rolünüzü değiştiremezsiniz')
        changes['role'] = (target['role'], data['role'])
        g.db.execute('UPDATE users SET role = ? WHERE id = ?', (data['role'], user_id))
    if 'is_active' in data and bool(data['is_active']) != bool(target['is_active']):
        if user_id == g.user['id']:
            return fail('Kendi hesabınızı devre dışı bırakamazsınız')
        changes['is_active'] = (bool(target['is_active']), bool(data['is_active']))
        g.db.execute('UPDATE users SET is_active = ? WHERE id = ?', (1 if data['is_active'] else 0, user_id))
    if data.get('unlock'):
        changes['unlocked'] = True
        g.db.execute('UPDATE users SET locked_until = NULL, failed_login_attempts = 0 WHERE id = ?', (user_id,))
    if data.get('password'):
        problem = security.password_problem(data['password'])
        if problem:
            return fail(problem)
        changes['password_reset'] = True
        g.db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (security.hash_password(data['password']), user_id))
    if not changes:
        return fail('Değişiklik yok')
    if 'role' in changes or 'is_active' in changes or 'password_reset' in changes:
        g.db.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))   # açık oturumları kapat
    audit('user.update', resource_type='user', resource_id=user_id,
          details={'username': target['username'], **{k: v for k, v in changes.items()}})
    g.db.commit()
    return ok({'message': 'Kullanıcı güncellendi'})


@app.route('/api/audit-logs')
@require('audit.view')
def audit_logs():
    clauses, params = [], []
    if request.args.get('username'):
        clauses.append('a.username = ?')
        params.append(request.args['username'])
    if request.args.get('action'):
        clauses.append('a.action LIKE ?')
        params.append(request.args['action'] + '%')
    if request.args.get('patient_id'):
        clauses.append('a.patient_id = ?')
        params.append(int(request.args['patient_id']))
    if request.args.get('failed') == '1':
        clauses.append('a.success = 0')
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    limit = min(int(request.args.get('limit', 100)), 500)
    offset = max(int(request.args.get('offset', 0)), 0)
    total = g.db.execute(f'SELECT COUNT(*) FROM audit_logs a {where}', params).fetchone()[0]
    rows = g.db.execute(
        f'''SELECT a.*, p.name AS patient_name FROM audit_logs a
            LEFT JOIN patients p ON p.id = a.patient_id {where}
            ORDER BY a.id DESC LIMIT ? OFFSET ?''', params + [limit, offset]).fetchall()
    logs = []
    for r in rows:
        d = dict(r)
        d['details'] = json.loads(d['details']) if d['details'] else None
        logs.append(d)
    return ok({'logs': logs, 'total': total})


@app.route('/api/admin/db-stats')
@require('audit.view')
def db_stats():
    return ok({'tables': db.stats(g.db), 'schema_version': db.SCHEMA_VERSION,
               'database': os.path.basename(db.DB_PATH), 'dataset': db.dataset_summary(g.db)})


@app.route('/api/admin/dataset.csv')
@require('dataset.export')
def export_dataset():
    """Etiketli, anonim veri seti (PhysioNet 2019 formatı) — yeniden eğitim için."""
    import csv
    import io
    header, rows = db.export_dataset_rows(g.db)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    audit('dataset.export', details={'rows': len(rows), 'patients': len({r[-1] for r in rows})})
    g.db.commit()
    from flask import Response
    filename = f"sepsisai_veriseti_{db.now()[:10]}.csv"
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


# ============================================================================
# MODEL VE SİSTEM
# ============================================================================

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy' if ml.is_ready() else 'degraded',
        'model_loaded': ml.model is not None,
        'preprocessing_loaded': ml.is_ready(),
        'feature_set': FEATURE_SET,
        'calibrator_loaded': ml.estimator is not None and ml.estimator.calibrator is not None,
        'database_exists': os.path.exists(db.DB_PATH),
    }), 200


@app.route('/api/model-info')
@require('model.view')
def model_info():
    return ok({
        'model': MODEL_INFO,
        'threshold': OPTIMAL_THRESHOLD,
        'window_size': ml.WINDOW_SIZE,
        'features': (ml.v4_scaler.columns if FEATURE_SET == 'v4' and ml.v4_scaler is not None
                     else feature_names() if FEATURE_SET == 'v3' else ml.numerical_columns),
        'feature_set': FEATURE_SET,
        'unused_fields': ml.MODEL_UNUSED_FIELDS,
        'reliability_stats': ml.estimator.stats if ml.estimator else {},
        'clinical_comparison': _read_model_json('clinical_comparison.json'),
        'cross_validation': _read_model_json('cross_validation.json'),
        'external_validation': _read_model_json('hastane_arasi_test.json'),
    })


def _read_model_json(name):
    path = os.path.join(os.path.dirname(ml.MODEL_PATH), name)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# BAŞLATMA
# ============================================================================

def init_app():
    """Veritabanını hazırlar, demo hesaplarını oluşturur."""
    db.init_database()
    conn = db.connect()
    security.ensure_demo_accounts(conn)
    conn.close()


def port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


if __name__ == '__main__':
    if port_in_use(PORT):
        print(f"\n[FAIL] {PORT} portu kullanimda. Uygulama zaten acik olabilir: http://localhost:{PORT}")
        print("       Eski pencereyi kapatip tekrar deneyin.")
        raise SystemExit(1)

    init_app()
    if not ml.load_model_and_preprocessing():
        print("\n[FAIL] Sunucu baslatilamadi. Model yukleme hatasi!")
        raise SystemExit(1)

    print("\n[...] Model warm-up yapiliyor...")
    try:
        ml.warm_up()
        print("  [OK] Model warm-up tamamlandi")
    except Exception as e:
        print(f"  [WARN] Warm-up hatasi (onemli degil): {e}")

    url = f"http://localhost:{PORT}"
    print("\n" + "=" * 60)
    print(f"[WEB] Web arayuzu: {url}")
    print(f"[INFO] Model: {MODEL_INFO.get('name')} ({FEATURE_SET}), esik {OPTIMAL_THRESHOLD}")
    if security.visible_demo_accounts():
        print("[INFO] Demo hesaplari: " + ", ".join(f"{a['username']} / {a['password']}" for a in security.visible_demo_accounts()))
    print("[INFO] Cikmak icin: CTRL+C")
    print("=" * 60 + "\n")

    if os.environ.get('SEPSIS_OPEN_BROWSER') == '1':
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(debug=False, host=HOST, port=PORT, threaded=True)
