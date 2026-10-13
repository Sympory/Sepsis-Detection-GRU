"""Veritabanı şeması, kısıtlar, CASCADE, denetim kaydının değiştirilemezliği ve v1 → v2 taşıma."""

import json
import sqlite3

import pytest


def insert_patient(conn, code='P1', **kw):
    cols = {'patient_id': code, 'name': 'Ad', **kw}
    q = f"INSERT INTO patients ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    return conn.execute(q, list(cols.values())).lastrowid


def test_all_tables_exist(fresh_db):
    conn = fresh_db.connect()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {'users', 'sessions', 'patients', 'observations', 'measurements', 'predictions',
            'alarms', 'outcomes', 'audit_logs'} <= tables


@pytest.mark.parametrize('kw', [{'age': 150}, {'age': -1}, {'gender': 'X'}, {'status': 'deleted'}])
def test_patient_check_constraints(fresh_db, kw):
    conn = fresh_db.connect()
    with pytest.raises(sqlite3.IntegrityError):
        insert_patient(conn, **kw)
    conn.close()


def test_unique_patient_code(fresh_db):
    conn = fresh_db.connect()
    insert_patient(conn, 'DUP')
    with pytest.raises(sqlite3.IntegrityError):
        insert_patient(conn, 'DUP')
    conn.close()


def test_invalid_role_rejected(fresh_db):
    conn = fresh_db.connect()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO users (username, password_hash, full_name, role) VALUES ('x', 'h', 'X', 'student')")
    conn.close()


def test_foreign_key_blocks_orphan_observation(fresh_db):
    conn = fresh_db.connect()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute('INSERT INTO observations (patient_id, hour) VALUES (999, 1)')
    conn.close()


def test_unique_patient_hour(fresh_db):
    conn = fresh_db.connect()
    pk = insert_patient(conn)
    conn.execute('INSERT INTO observations (patient_id, hour) VALUES (?, 1)', (pk,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute('INSERT INTO observations (patient_id, hour) VALUES (?, 1)', (pk,))
    conn.close()


def test_cascade_delete(fresh_db):
    conn = fresh_db.connect()
    pk = insert_patient(conn)
    obs = conn.execute('INSERT INTO observations (patient_id, hour) VALUES (?, 1)', (pk,)).lastrowid
    conn.execute("INSERT INTO measurements VALUES (?, 'HR', 90)", (obs,))
    conn.execute("INSERT INTO predictions (observation_id, score, risk_level) VALUES (?, 0.2, 'Orta')", (obs,))
    conn.execute("INSERT INTO alarms (patient_id, observation_id, risk_level) VALUES (?, ?, 'Yüksek')", (pk, obs))
    conn.execute("INSERT INTO outcomes (patient_id, sepsis) VALUES (?, 'no')", (pk,))
    conn.execute('DELETE FROM patients WHERE id = ?', (pk,))
    for table in ('observations', 'measurements', 'predictions', 'alarms', 'outcomes'):
        assert conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0, table
    conn.close()


def test_outcome_onset_only_for_sepsis(fresh_db):
    conn = fresh_db.connect()
    pk = insert_patient(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO outcomes (patient_id, sepsis, onset_hour) VALUES (?, 'no', 5)", (pk,))
    conn.execute("INSERT INTO outcomes (patient_id, sepsis, onset_hour) VALUES (?, 'yes', 5)", (pk,))
    conn.close()


def test_audit_log_is_immutable(fresh_db):
    conn = fresh_db.connect()
    fresh_db.audit(conn, None, 'test.event')
    conn.commit()
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit_logs SET action = 'degisti'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute('DELETE FROM audit_logs')
    conn.close()


def test_migration_from_v1(tmp_path, monkeypatch):
    """Eski sürümün JSON tabanlı veritabanı yedeklenip normalize şemaya taşınmalı."""
    import db
    path = tmp_path / 'v1.db'
    old = sqlite3.connect(path)
    old.executescript('''
        CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                               age INTEGER, gender TEXT, admission_time TEXT, created_at TEXT);
        CREATE TABLE hourly_data (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, hour INTEGER NOT NULL,
                                  vital_signs TEXT NOT NULL, prediction REAL, risk_level TEXT, timestamp TEXT,
                                  reliability REAL, assessment TEXT, UNIQUE(patient_id, hour));
    ''')
    old.execute("INSERT INTO patients (patient_id, name, age, gender) VALUES ('P1', 'Eski Hasta', 60, 'Male')")
    old.execute("INSERT INTO hourly_data (patient_id, hour, vital_signs, prediction, risk_level, reliability) VALUES "
                "(1, 1, ?, 0.4, 'Yüksek', 80)", (json.dumps({'HR': 120, 'Temp': 38.9}),))
    old.commit()
    old.close()

    monkeypatch.setattr(db, 'DB_PATH', str(path))
    db.init_database()
    conn = db.connect()
    assert conn.execute('PRAGMA user_version').fetchone()[0] == db.SCHEMA_VERSION
    obs = db.get_observations(conn, 1)
    assert obs[0]['vital_signs'] == {'HR': 120.0, 'Temp': 38.9}
    assert obs[0]['risk_level'] == 'Yüksek'
    assert conn.execute("SELECT COUNT(*) FROM alarms WHERE status = 'open'").fetchone()[0] == 1   # son risk yüksek
    assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    conn.close()
    assert list(tmp_path.glob('v1.db.yedek-*')), 'Taşımadan önce yedek alınmalı'
