"""
Sepsis Tahmin Sistemi — Veritabanı Katmanı (SQLite)
===================================================

Şema (normalize):

    users ──< sessions
      │
      ├──< audit_logs            (değiştirilemez: UPDATE/DELETE tetikleyiciyle engellenir)
      │
    patients ──< observations ──< measurements      (her değer ayrı satır)
       │              │
       │              └──── predictions             (1-1: skor, risk, güvenilirlik, model sürümü)
       ├──< alarms                                  (yüksek risk uyarıları, doktor onayı)
       └──── outcomes                               (1-1: doktorun işaretlediği sepsis sonucu → veri seti etiketi)

Eski sürümün `hourly_data` tablosu (ölçümler JSON metin) ilk açılışta yedek
alınarak bu yapıya taşınır ve `legacy_hourly_data` adıyla saklanır.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime

from config import DB_PATH as DEFAULT_DB_PATH

SCHEMA_VERSION = 3   # 1: hourly_data (JSON) · 2: normalize şema · 3: + outcomes
DB_PATH = DEFAULT_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    username               TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash          TEXT    NOT NULL,
    full_name              TEXT    NOT NULL,
    role                   TEXT    NOT NULL CHECK (role IN ('admin', 'doctor', 'nurse')),
    is_active              INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    failed_login_attempts  INTEGER NOT NULL DEFAULT 0,
    locked_until           TEXT,
    last_login             TEXT,
    created_at             TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    created_by             INTEGER REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash     TEXT PRIMARY KEY,                 -- çerezdeki token'ın SHA-256 özeti
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address     TEXT,
    user_agent     TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_activity  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS audit_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    user_id        INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username       TEXT,                             -- kullanıcı silinse de kim olduğu kalsın
    role           TEXT,
    action         TEXT    NOT NULL,
    resource_type  TEXT,
    resource_id    INTEGER,
    patient_id     INTEGER,                          -- FK yok: hasta silinse de kayıt kalsın
    details        TEXT,                             -- JSON (ör. eski/yeni değerler)
    ip_address     TEXT,
    success        INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_audit_time    ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user    ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_action  ON audit_logs(action);

CREATE TRIGGER IF NOT EXISTS audit_logs_no_update BEFORE UPDATE ON audit_logs
BEGIN SELECT RAISE(ABORT, 'Denetim kayıtları değiştirilemez'); END;
CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete BEFORE DELETE ON audit_logs
BEGIN SELECT RAISE(ABORT, 'Denetim kayıtları silinemez'); END;

CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT    NOT NULL UNIQUE,         -- hastane protokol numarası
    name            TEXT    NOT NULL,
    age             INTEGER CHECK (age IS NULL OR age BETWEEN 0 AND 120),
    gender          TEXT    CHECK (gender IS NULL OR gender IN ('Male', 'Female')),
    admission_time  TEXT,
    status          TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'discharged')),
    discharged_at   TEXT,
    discharged_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_patients_status ON patients(status);

CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    hour        INTEGER NOT NULL CHECK (hour >= 0),  -- YBÜ'ye girişten bu yana saat
    entered_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    entered_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at  TEXT,
    UNIQUE (patient_id, hour)
);
CREATE INDEX IF NOT EXISTS idx_obs_patient_hour ON observations(patient_id, hour);

CREATE TABLE IF NOT EXISTS measurements (
    observation_id  INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    variable        TEXT    NOT NULL,                -- HR, Temp, Lactate, ...
    value           REAL    NOT NULL,
    PRIMARY KEY (observation_id, variable)
);
CREATE INDEX IF NOT EXISTS idx_meas_variable ON measurements(variable);

CREATE TABLE IF NOT EXISTS predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id          INTEGER NOT NULL UNIQUE REFERENCES observations(id) ON DELETE CASCADE,
    model_version           TEXT,
    score                   REAL CHECK (score IS NULL OR score BETWEEN 0 AND 1),
    calibrated_probability  REAL,
    risk_level              TEXT,
    reliability             REAL CHECK (reliability IS NULL OR reliability BETWEEN 0 AND 100),
    assessment              TEXT,                    -- JSON: bileşenler, gerekçeler, MC-Dropout
    created_at              TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS alarms (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    observation_id   INTEGER REFERENCES observations(id) ON DELETE SET NULL,
    risk_level       TEXT    NOT NULL,
    score            REAL,
    status           TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged')),
    created_at       TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    acknowledged_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at  TEXT,
    note             TEXT
);
CREATE INDEX IF NOT EXISTS idx_alarms_open ON alarms(status, patient_id);

-- Doktorun taburcuda işaretlediği gerçek sonuç: yeniden eğitim veri setinin etiketi
CREATE TABLE IF NOT EXISTS outcomes (
    patient_id   INTEGER PRIMARY KEY REFERENCES patients(id) ON DELETE CASCADE,   -- PK = FK: hasta başına tek sonuç
    sepsis       TEXT    NOT NULL CHECK (sepsis IN ('yes', 'no', 'unknown')),
    onset_hour   INTEGER CHECK (onset_hour IS NULL OR onset_hour >= 0),           -- klinik sepsis başlangıcı (YBÜ saati)
    note         TEXT,
    labeled_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    labeled_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    CHECK (sepsis = 'yes' OR onset_hour IS NULL)                                   -- başlangıç saati yalnızca sepsiste
);

-- Her hastanın en son saatlik kaydı ve tahmini
CREATE VIEW IF NOT EXISTS v_patient_latest AS
SELECT p.id AS patient_pk,
       o.id AS observation_id, o.hour, o.entered_at,
       pr.score, pr.risk_level, pr.reliability
FROM patients p
JOIN observations o ON o.id = (SELECT id FROM observations WHERE patient_id = p.id ORDER BY hour DESC LIMIT 1)
LEFT JOIN predictions pr ON pr.observation_id = o.id;
"""


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ============================================================================
# KURULUM VE TAŞIMA
# ============================================================================

def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn, table):
    return {r[1] for r in conn.execute(f'PRAGMA table_info({table})')}


def init_database():
    """Şemayı oluşturur; eski sürüm veritabanı varsa yedek alıp taşır."""
    needs_migration = False
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        needs_migration = version < 2 and _table_exists(conn, 'patients')   # yalnızca v1 → v2 veri taşıması
        conn.close()

    if needs_migration:
        backup = f"{DB_PATH}.yedek-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(DB_PATH, backup)
        print(f"[OK] Eski veritabani yedeklendi: {os.path.basename(backup)}")
        try:
            _migrate_v1(backup_name=os.path.basename(backup))
        except Exception:
            shutil.copy2(backup, DB_PATH)
            print("[FAIL] Tasima basarisiz, yedek geri yuklendi")
            raise

    conn = connect()
    conn.execute('PRAGMA journal_mode = WAL')
    conn.executescript(SCHEMA)
    conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
    conn.commit()
    conn.close()
    print("[OK] Veritabani hazir (sema surumu %d)" % SCHEMA_VERSION)


def _to_age(value):
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    return age if 0 <= age <= 120 else None


def _migrate_v1(backup_name):
    """v1 (patients + hourly_data JSON) → v2 normalize şema"""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = OFF')
    try:
        # Yeni tablolar eski patients'ın yerini alacak şekilde oluşturulur
        old_cols = _columns(conn, 'patients')
        conn.execute('ALTER TABLE patients RENAME TO patients_v1')
        conn.executescript(SCHEMA)

        conn.execute('BEGIN')
        for p in conn.execute('SELECT * FROM patients_v1').fetchall():
            status = p['status'] if 'status' in old_cols and p['status'] in ('active', 'discharged') else 'active'
            conn.execute(
                '''INSERT INTO patients (id, patient_id, name, age, gender, admission_time, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (p['id'], p['patient_id'], p['name'], _to_age(p['age']),
                 p['gender'] if p['gender'] in ('Male', 'Female') else None,
                 p['admission_time'], status, p['created_at'] or now()))
        conn.execute('DROP TABLE patients_v1')

        # hourly_data (JSON) → observations + measurements + predictions
        moved = 0
        if _table_exists(conn, 'hourly_data'):
            hcols = _columns(conn, 'hourly_data')
            for h in conn.execute('SELECT * FROM hourly_data ORDER BY patient_id, hour').fetchall():
                obs_id = conn.execute('INSERT INTO observations (patient_id, hour, entered_at) VALUES (?, ?, ?)',
                                      (h['patient_id'], h['hour'], h['timestamp'] or now())).lastrowid
                try:
                    values = json.loads(h['vital_signs'] or '{}')
                except ValueError:
                    values = {}
                for var, val in values.items():
                    try:
                        conn.execute('INSERT INTO measurements VALUES (?, ?, ?)', (obs_id, var, float(val)))
                    except (TypeError, ValueError):
                        pass
                assessment = h['assessment'] if 'assessment' in hcols else None
                calibrated = None
                if assessment:
                    try:
                        calibrated = json.loads(assessment).get('calibrated_probability')
                    except ValueError:
                        pass
                conn.execute(
                    '''INSERT INTO predictions (observation_id, model_version, score, calibrated_probability,
                                               risk_level, reliability, assessment, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (obs_id, 'v1 (taşındı)', h['prediction'], calibrated, h['risk_level'],
                     h['reliability'] if 'reliability' in hcols else None, assessment, h['timestamp'] or now()))
                moved += 1
            # Eski tablo arşivlenir; kısıtsız kopya (eski yabancı anahtar artık geçersiz)
            conn.execute('CREATE TABLE legacy_hourly_data AS SELECT * FROM hourly_data')
            conn.execute('DROP TABLE hourly_data')

        # Taşınan verinin son riski yüksekse açık alarm oluştur (yeni sürümdeki davranışla tutarlı)
        alarms = 0
        for row in conn.execute('SELECT patient_pk, observation_id, risk_level, score FROM v_patient_latest').fetchall():
            if row['risk_level'] and 'yüksek' in row['risk_level'].lower():
                conn.execute('INSERT INTO alarms (patient_id, observation_id, risk_level, score, created_at) VALUES (?, ?, ?, ?, ?)',
                             (row['patient_pk'], row['observation_id'], row['risk_level'], row['score'], now()))
                alarms += 1

        conn.execute(
            "INSERT INTO audit_logs (timestamp, username, role, action, details) VALUES (?, 'sistem', 'system', 'db.migrate', ?)",
            (now(), json.dumps({'from': 1, 'to': 2, 'observations': moved, 'alarms': alarms, 'backup': backup_name}),))
        conn.execute('COMMIT')
        conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
        print(f"[OK] Veritabani tasindi: {moved} saatlik kayit normalize tablolara aktarildi")
    finally:
        conn.close()


# ============================================================================
# DENETİM KAYDI
# ============================================================================

def audit(conn, user, action, *, resource_type=None, resource_id=None, patient_id=None,
          details=None, ip=None, success=True):
    conn.execute(
        '''INSERT INTO audit_logs (timestamp, user_id, username, role, action, resource_type, resource_id,
                                   patient_id, details, ip_address, success)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (now(), user['id'] if user else None, user['username'] if user else None, user['role'] if user else None,
         action, resource_type, resource_id, patient_id,
         json.dumps(details, ensure_ascii=False) if details is not None else None, ip, 1 if success else 0))


# ============================================================================
# HASTA VERİSİ
# ============================================================================

def list_patients(conn, status='active'):
    rows = conn.execute(
        '''SELECT p.*,
                  (SELECT COUNT(*) FROM observations o WHERE o.patient_id = p.id) AS total_hours,
                  l.hour AS latest_hour, l.score AS latest_prediction, l.risk_level AS latest_risk_level,
                  l.reliability AS latest_reliability, l.entered_at AS latest_timestamp,
                  (SELECT COUNT(*) FROM alarms a WHERE a.patient_id = p.id AND a.status = 'open') AS open_alarms,
                  oc.sepsis AS outcome_sepsis
           FROM patients p
           LEFT JOIN v_patient_latest l ON l.patient_pk = p.id
           LEFT JOIN outcomes oc ON oc.patient_id = p.id
           WHERE (? = 'all' OR p.status = ?)
           ORDER BY p.created_at DESC''', (status, status)).fetchall()
    return [{
        'id': p['id'], 'patient_id': p['patient_id'], 'name': p['name'], 'age': p['age'],
        'gender': p['gender'], 'admission_time': p['admission_time'], 'status': p['status'],
        'discharged_at': p['discharged_at'], 'total_hours': p['total_hours'],
        'hourly_data_count': p['total_hours'], 'latest_hour': p['latest_hour'],
        'latest_prediction': p['latest_prediction'], 'latest_risk': p['latest_risk_level'],
        'latest_risk_level': p['latest_risk_level'], 'latest_reliability': p['latest_reliability'],
        'latest_timestamp': p['latest_timestamp'], 'open_alarms': p['open_alarms'],
        'outcome_sepsis': p['outcome_sepsis'],
        'created_at': p['created_at'],
    } for p in rows]


def get_patient(conn, patient_pk):
    p = conn.execute(
        '''SELECT p.*, uc.full_name AS created_by_name, ud.full_name AS discharged_by_name
           FROM patients p
           LEFT JOIN users uc ON uc.id = p.created_by
           LEFT JOIN users ud ON ud.id = p.discharged_by
           WHERE p.id = ?''', (patient_pk,)).fetchone()
    if not p:
        return None
    patient = dict(p)
    oc = conn.execute('''SELECT oc.*, u.full_name AS labeled_by_name FROM outcomes oc
                         LEFT JOIN users u ON u.id = oc.labeled_by WHERE oc.patient_id = ?''', (patient_pk,)).fetchone()
    patient['outcome'] = dict(oc) if oc else None
    return patient


def save_outcome(conn, patient_pk, sepsis, onset_hour, note, user):
    """Hastanın sonucunu yazar (varsa günceller); eski değeri döndürür."""
    old = conn.execute('SELECT sepsis, onset_hour, note FROM outcomes WHERE patient_id = ?', (patient_pk,)).fetchone()
    conn.execute(
        '''INSERT INTO outcomes (patient_id, sepsis, onset_hour, note, labeled_by, labeled_at) VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(patient_id) DO UPDATE SET sepsis = excluded.sepsis, onset_hour = excluded.onset_hour,
               note = excluded.note, labeled_by = excluded.labeled_by, labeled_at = excluded.labeled_at''',
        (patient_pk, sepsis, onset_hour, note, user['id'], now()))
    return dict(old) if old else None


# ============================================================================
# VERİ SETİ (yeniden eğitim için, PhysioNet 2019 formatında)
# ============================================================================

def dataset_summary(conn):
    row = conn.execute(
        '''SELECT
             SUM(CASE WHEN oc.sepsis = 'yes' THEN 1 ELSE 0 END) AS sepsis_yes,
             SUM(CASE WHEN oc.sepsis = 'no' THEN 1 ELSE 0 END) AS sepsis_no,
             SUM(CASE WHEN oc.sepsis = 'unknown' THEN 1 ELSE 0 END) AS sepsis_unknown,
             SUM(CASE WHEN oc.patient_id IS NULL THEN 1 ELSE 0 END) AS unlabeled
           FROM patients p LEFT JOIN outcomes oc ON oc.patient_id = p.id
           WHERE p.status = 'discharged' ''').fetchone()
    hours = conn.execute(
        '''SELECT COUNT(*) FROM observations o JOIN outcomes oc ON oc.patient_id = o.patient_id
           JOIN patients p ON p.id = o.patient_id
           WHERE p.status = 'discharged' AND oc.sepsis IN ('yes', 'no')''').fetchone()[0]
    d = {k: (row[k] or 0) for k in ('sepsis_yes', 'sepsis_no', 'sepsis_unknown', 'unlabeled')}
    d['exportable_patients'] = d['sepsis_yes'] + d['sepsis_no']
    d['exportable_hours'] = hours
    return d


def export_dataset_rows(conn, label_lead_hours=6):
    """
    Taburcu edilmiş ve sonucu 'yes'/'no' olarak işaretlenmiş hastaların saatlik kayıtları.
    Kimlik bilgisi yok: ad, protokol no ve iç id yerine sıra numarası verilir.
    SepsisLabel, PhysioNet 2019 kuralıyla klinik başlangıçtan `label_lead_hours` saat önce 1 olur.
    """
    from features import CLINICAL_COLUMNS
    patients = conn.execute(
        '''SELECT p.id, p.age, p.gender, oc.sepsis, oc.onset_hour FROM patients p
           JOIN outcomes oc ON oc.patient_id = p.id
           WHERE p.status = 'discharged' AND oc.sepsis IN ('yes', 'no') ORDER BY p.id''').fetchall()
    header = ['Hour', *CLINICAL_COLUMNS, 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'Patient_ID']
    rows = []
    for anon_id, p in enumerate(patients, start=1):
        for o in get_observations(conn, p['id']):
            vs = o['vital_signs']
            label = int(p['sepsis'] == 'yes' and p['onset_hour'] is not None and o['hour'] >= p['onset_hour'] - label_lead_hours)
            rows.append([o['hour'], *[vs.get(c, '') for c in CLINICAL_COLUMNS], p['age'] if p['age'] is not None else '',
                         {'Male': 1, 'Female': 0}.get(p['gender'], ''), '', '', '', o['hour'] + 1, label, f'L{anon_id:05d}'])
    return header, rows


def get_observations(conn, patient_pk, before_hour=None):
    """Hastanın saatlik kayıtları: ölçümler, tahmin ve kaydı giren kişi ile"""
    obs = conn.execute(
        '''SELECT o.id, o.hour, o.entered_at, o.updated_at, u.full_name AS entered_by_name, u.role AS entered_by_role,
                  pr.score, pr.risk_level, pr.reliability, pr.assessment, pr.model_version
           FROM observations o
           LEFT JOIN users u ON u.id = o.entered_by
           LEFT JOIN predictions pr ON pr.observation_id = o.id
           WHERE o.patient_id = ? AND (? IS NULL OR o.hour < ?)
           ORDER BY o.hour''', (patient_pk, before_hour, before_hour)).fetchall()
    values = {}
    if obs:
        ids = [o['id'] for o in obs]
        q = f"SELECT observation_id, variable, value FROM measurements WHERE observation_id IN ({','.join('?' * len(ids))})"
        for m in conn.execute(q, ids):
            values.setdefault(m['observation_id'], {})[m['variable']] = m['value']
    return [{
        'id': o['id'], 'hour': o['hour'], 'vital_signs': values.get(o['id'], {}),
        'prediction': o['score'], 'risk_level': o['risk_level'], 'reliability': o['reliability'],
        'assessment': json.loads(o['assessment']) if o['assessment'] else None,
        'model_version': o['model_version'], 'timestamp': o['updated_at'] or o['entered_at'],
        'entered_by': o['entered_by_name'], 'entered_by_role': o['entered_by_role'],
    } for o in obs]


def save_observation(conn, patient_pk, hour, values, user, result, model_version):
    """
    Saatlik kaydı ölçümler + tahminle birlikte yazar. Aynı saat zaten varsa üzerine
    yazar ve eski değerleri döndürür (denetim kaydına yazılmak üzere).
    """
    existing = conn.execute('SELECT id FROM observations WHERE patient_id = ? AND hour = ?',
                            (patient_pk, hour)).fetchone()
    old_values = None
    if existing:
        obs_id = existing['id']
        old_values = {r['variable']: r['value'] for r in
                      conn.execute('SELECT variable, value FROM measurements WHERE observation_id = ?', (obs_id,))}
        conn.execute('DELETE FROM measurements WHERE observation_id = ?', (obs_id,))
        conn.execute('DELETE FROM predictions WHERE observation_id = ?', (obs_id,))
        conn.execute('UPDATE observations SET entered_by = ?, updated_at = ? WHERE id = ?', (user['id'], now(), obs_id))
    else:
        obs_id = conn.execute('INSERT INTO observations (patient_id, hour, entered_by, entered_at) VALUES (?, ?, ?, ?)',
                              (patient_pk, hour, user['id'], now())).lastrowid

    conn.executemany('INSERT INTO measurements (observation_id, variable, value) VALUES (?, ?, ?)',
                     [(obs_id, k, float(v)) for k, v in values.items()])
    a = result['assessment']
    conn.execute(
        '''INSERT INTO predictions (observation_id, model_version, score, calibrated_probability,
                                    risk_level, reliability, assessment, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (obs_id, model_version, result['prediction'], a.get('calibrated_probability'), result['risk_level'],
         a['score'], json.dumps(a, ensure_ascii=False), now()))
    return obs_id, old_values


def raise_alarm_if_needed(conn, patient_pk, obs_id, result):
    """Yüksek/Çok Yüksek riskte, hastanın açık alarmı yoksa yeni alarm açar."""
    risk = (result['risk_level'] or '').lower()
    if not result['assessment']['sufficient_data'] or 'yüksek' not in risk:
        return None
    if conn.execute("SELECT 1 FROM alarms WHERE patient_id = ? AND status = 'open'", (patient_pk,)).fetchone():
        return None
    return conn.execute('INSERT INTO alarms (patient_id, observation_id, risk_level, score, created_at) VALUES (?, ?, ?, ?, ?)',
                        (patient_pk, obs_id, result['risk_level'], result['prediction'], now())).lastrowid


def list_alarms(conn, status='open'):
    rows = conn.execute(
        '''SELECT a.*, p.name AS patient_name, p.patient_id AS patient_code, p.status AS patient_status,
                  o.hour, u.full_name AS acknowledged_by_name
           FROM alarms a
           JOIN patients p ON p.id = a.patient_id
           LEFT JOIN observations o ON o.id = a.observation_id
           LEFT JOIN users u ON u.id = a.acknowledged_by
           WHERE (? = 'all' OR a.status = ?)
           ORDER BY a.created_at DESC LIMIT 200''', (status, status)).fetchall()
    return [dict(r) for r in rows]


def patient_history(conn, patient_pk, limit=200):
    rows = conn.execute(
        '''SELECT timestamp, username, role, action, details, success
           FROM audit_logs WHERE patient_id = ? ORDER BY id DESC LIMIT ?''', (patient_pk, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d['details'] = json.loads(d['details']) if d['details'] else None
        out.append(d)
    return out


def stats(conn):
    """Yönetici paneli: tablo başına kayıt sayısı"""
    tables = ['users', 'patients', 'observations', 'measurements', 'predictions', 'alarms', 'outcomes', 'audit_logs', 'sessions']
    return {t: conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in tables}
