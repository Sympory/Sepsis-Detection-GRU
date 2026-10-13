"""Giriş, oturum, şifre ve rol bazlı yetki testleri."""

from datetime import datetime, timedelta

import pytest

from conftest import login


def test_password_hash_not_plaintext(fresh_db):
    import security
    h = security.hash_password('Guclu1234')
    assert 'Guclu1234' not in h
    assert security.verify_password('Guclu1234', h)
    assert not security.verify_password('yanlis', h)
    assert security.hash_password('Guclu1234') != h   # tuz: aynı şifrenin hash'i farklı


@pytest.mark.parametrize('password,ok', [('kisa1', False), ('sadeceharf', False), ('12345678', False), ('Guclu1234', True)])
def test_password_policy(password, ok):
    import security
    assert (security.password_problem(password) is None) == ok


def test_login_and_me(app_module):
    client = login(app_module, 'doctor')
    me = client.get('/api/auth/me').get_json()
    assert me['user']['role'] == 'doctor'
    assert 'alarms.acknowledge' in me['user']['permissions']


def test_unauthenticated_requests_blocked(app_module):
    anon = app_module.app.test_client()
    assert anon.get('/').status_code == 302
    assert anon.get('/api/patients').status_code == 401
    assert anon.get('/patients.db').status_code == 404
    assert anon.get('/config.py').status_code == 404


def test_lockout_after_five_failures(app_module, fresh_db):
    client = app_module.app.test_client()
    for _ in range(5):
        r = client.post('/api/auth/login', json={'username': 'doktor', 'password': 'yanlis'})
        assert r.status_code == 401
    assert 'kilit' in r.get_json()['error']
    r = client.post('/api/auth/login', json={'username': 'doktor', 'password': 'Doktor123'})
    assert r.status_code == 401, 'Doğru şifre de kilit süresince reddedilmeli'


def test_session_idle_timeout(app_module, fresh_db):
    client = login(app_module, 'nurse')
    conn = fresh_db.connect()
    old = (datetime.now() - timedelta(minutes=31)).strftime('%Y-%m-%d %H:%M:%S')
    conn.execute('UPDATE sessions SET last_activity = ?', (old,))
    conn.commit()
    conn.close()
    assert client.get('/api/patients').status_code == 401


def test_session_token_stored_hashed(app_module, fresh_db):
    client = login(app_module, 'nurse')
    token = client.get_cookie('sepsis_session').value
    conn = fresh_db.connect()
    stored = [r[0] for r in conn.execute('SELECT token_hash FROM sessions')]
    conn.close()
    assert token not in stored and len(stored[0]) == 64


def test_logout_ends_session(app_module):
    client = login(app_module, 'doctor')
    assert client.post('/api/auth/logout').status_code == 200
    assert client.get('/api/patients').status_code == 401


# (rol, yöntem, yol, beklenen) — yetkisiz istekler 403, yetkililer 403 olmamalı
PERMISSION_CASES = [
    ('nurse', 'post', '/api/users', 403),
    ('doctor', 'post', '/api/users', 403),
    ('nurse', 'get', '/api/audit-logs', 403),
    ('doctor', 'get', '/api/audit-logs', 403),
    ('admin', 'get', '/api/audit-logs', 200),
    ('nurse', 'get', '/api/admin/dataset.csv', 403),
    ('doctor', 'get', '/api/admin/dataset.csv', 403),
    ('admin', 'get', '/api/admin/dataset.csv', 200),
    ('admin', 'post', '/api/patients', 403),
]


@pytest.mark.parametrize('role,method,path,expected', PERMISSION_CASES)
def test_role_permissions(clients, role, method, path, expected):
    r = getattr(clients[role], method)(path, json={})
    assert r.status_code == expected


def test_nurse_cannot_discharge_or_acknowledge(clients, patient):
    assert clients['nurse'].post(f'/api/patients/{patient}/discharge', json={}).status_code == 403
    assert clients['nurse'].put(f'/api/patients/{patient}/outcome', json={}).status_code == 403
    assert clients['nurse'].post('/api/alarms/1/acknowledge', json={'note': 'x'}).status_code == 403


def test_doctor_cannot_delete_admin_can(clients, patient):
    assert clients['doctor'].delete(f'/api/patients/{patient}').status_code == 403
    assert clients['admin'].delete(f'/api/patients/{patient}').status_code == 200


def test_admin_cannot_demote_self(clients):
    me = clients['admin'].get('/api/auth/me').get_json()['user']
    r = clients['admin'].patch(f"/api/users/{me['id']}", json={'role': 'nurse'})
    assert r.status_code == 400


def test_forbidden_attempt_is_audited(clients, fresh_db):
    clients['nurse'].get('/api/audit-logs')
    conn = fresh_db.connect()
    n = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE action = 'auth.forbidden' AND success = 0").fetchone()[0]
    conn.close()
    assert n >= 1
