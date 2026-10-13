"""
Sepsis Tahmin Sistemi — Kimlik Doğrulama ve Yetkilendirme
=========================================================

- Şifreler Werkzeug ile tuzlu (salted) hash olarak saklanır, düz metin asla tutulmaz.
- Oturum: rastgele token HttpOnly çerezde; veritabanında yalnızca SHA-256 özeti tutulur.
- 30 dk işlem yapılmazsa oturum düşer; 5 hatalı girişte hesap 15 dk kilitlenir.
- Yetkiler rol bazlıdır (PERMISSIONS); her API ucu bir izin ister.
"""

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta
from functools import wraps

from flask import g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

import db

SESSION_COOKIE = 'sepsis_session'
SESSION_IDLE_MINUTES = 30
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

ROLE_LABELS = {'admin': 'Yönetici', 'doctor': 'Doktor', 'nurse': 'Hemşire'}

ALL = {'admin', 'doctor', 'nurse'}
CLINICAL = {'doctor', 'nurse'}
PERMISSIONS = {
    'patients.view':        ALL,
    'patients.create':      CLINICAL,
    'observations.create':  CLINICAL,
    'patients.discharge':   {'doctor'},
    'outcomes.label':       {'doctor'},
    'dataset.export':       {'admin'},
    'patients.delete':      {'admin'},
    'alarms.view':          ALL,
    'alarms.acknowledge':   {'doctor'},
    'history.view':         ALL,
    'model.view':           ALL,
    'users.manage':         {'admin'},
    'audit.view':           {'admin'},
}

# Sunum/geliştirme için hazır hesaplar. Gerçek kullanımda SEPSIS_DEMO_ACCOUNTS=0 ile kapatılır.
DEMO_ACCOUNTS = [
    {'username': 'yonetici', 'password': 'Yonetici123', 'full_name': 'Sistem Yöneticisi', 'role': 'admin'},
    {'username': 'doktor', 'password': 'Doktor123', 'full_name': 'Dr. Elif Yılmaz', 'role': 'doctor'},
    {'username': 'hemsire', 'password': 'Hemsire123', 'full_name': 'Hemşire Can Aydın', 'role': 'nurse'},
]


def demo_enabled():
    return os.environ.get('SEPSIS_DEMO_ACCOUNTS', '1') != '0'


def public_demo():
    """İnternette herkese açık demo (ör. Hugging Face): yönetici gizli, uyarı bandı açık."""
    return os.environ.get('SEPSIS_PUBLIC_DEMO') == '1'


def visible_demo_accounts():
    """Giriş ekranında gösterilen hesaplar. Herkese açık demoda yönetici hesabı gösterilmez."""
    if not demo_enabled():
        return []
    return [a for a in DEMO_ACCOUNTS if not (public_demo() and a['role'] == 'admin')]


def _demo_password(acc):
    # Herkese açık demoda yönetici şifresi gizli ortam değişkeninden gelir; yoksa kimsenin bilmediği rastgele şifre
    if public_demo() and acc['role'] == 'admin':
        return os.environ.get('SEPSIS_ADMIN_PASSWORD') or secrets.token_urlsafe(24)
    return acc['password']


# ============================================================================
# ŞİFRE
# ============================================================================

def hash_password(password):
    return generate_password_hash(password)


def verify_password(password, password_hash):
    return check_password_hash(password_hash, password)


def password_problem(password):
    """Şifre kuralı: en az 8 karakter, en az bir harf ve bir rakam. Sorun yoksa None."""
    if len(password or '') < 8:
        return 'Şifre en az 8 karakter olmalı'
    if not re.search(r'[A-Za-zÇĞİÖŞÜçğıöşü]', password) or not re.search(r'\d', password):
        return 'Şifre en az bir harf ve bir rakam içermeli'
    return None


def ensure_demo_accounts(conn):
    """Hiç kullanıcı yoksa demo hesaplarını oluşturur (yalnızca demo modunda)."""
    if not demo_enabled() or conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] > 0:
        return False
    for acc in DEMO_ACCOUNTS:
        conn.execute('INSERT INTO users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)',
                     (acc['username'], hash_password(_demo_password(acc)), acc['full_name'], acc['role']))
    db.audit(conn, None, 'users.demo_created', details={'users': [a['username'] for a in DEMO_ACCOUNTS]})
    conn.commit()
    print("[OK] Demo hesaplari olusturuldu: " + ", ".join(a['username'] for a in DEMO_ACCOUNTS))
    return True


# ============================================================================
# OTURUM
# ============================================================================

def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _parse(ts):
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if ts else None


def client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()


def public_user(row):
    return {'id': row['id'], 'username': row['username'], 'full_name': row['full_name'],
            'role': row['role'], 'role_label': ROLE_LABELS.get(row['role'], row['role']),
            'permissions': sorted(p for p, roles in PERMISSIONS.items() if row['role'] in roles)}


def authenticate(conn, username, password):
    """
    Kullanıcı adı/şifre kontrolü. Dönüş: (kullanıcı satırı | None, hata mesajı | None).
    Başarısız denemeler sayılır, sınırı aşan hesap kilitlenir.
    """
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if not user:
        return None, 'Kullanıcı adı veya şifre hatalı'
    if not user['is_active']:
        return None, 'Hesap devre dışı. Yöneticiye başvurun.'
    locked_until = _parse(user['locked_until'])
    if locked_until and locked_until > datetime.now():
        minutes = int((locked_until - datetime.now()).total_seconds() // 60) + 1
        return None, f'Çok fazla hatalı deneme. Hesap {minutes} dakika kilitli.'

    if not verify_password(password, user['password_hash']):
        attempts = user['failed_login_attempts'] + 1
        if attempts >= MAX_LOGIN_ATTEMPTS:
            until = (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            conn.execute('UPDATE users SET failed_login_attempts = 0, locked_until = ? WHERE id = ?', (until, user['id']))
            db.audit(conn, dict(user), 'auth.locked', resource_type='user', resource_id=user['id'],
                     ip=client_ip(), success=False, details={'minutes': LOCKOUT_MINUTES})
            return None, f'Çok fazla hatalı deneme. Hesap {LOCKOUT_MINUTES} dakika kilitlendi.'
        conn.execute('UPDATE users SET failed_login_attempts = ? WHERE id = ?', (attempts, user['id']))
        left = MAX_LOGIN_ATTEMPTS - attempts
        return None, f'Kullanıcı adı veya şifre hatalı ({left} deneme hakkı kaldı)'

    conn.execute('UPDATE users SET failed_login_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?',
                 (db.now(), user['id']))
    return user, None


def create_session(conn, user):
    token = secrets.token_urlsafe(32)
    conn.execute('INSERT INTO sessions (token_hash, user_id, ip_address, user_agent, created_at, last_activity) '
                 'VALUES (?, ?, ?, ?, ?, ?)',
                 (_token_hash(token), user['id'], client_ip(), (request.user_agent.string or '')[:200], db.now(), db.now()))
    return token


def end_session(conn, token):
    if token:
        conn.execute('DELETE FROM sessions WHERE token_hash = ?', (_token_hash(token),))


def load_user(conn):
    """Çerezdeki oturumu doğrular; geçerliyse kullanıcıyı döndürür ve son etkinliği günceller."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    th = _token_hash(token)
    row = conn.execute(
        '''SELECT u.*, s.last_activity FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = ?''', (th,)).fetchone()
    if not row:
        return None
    if not row['is_active'] or _parse(row['last_activity']) < datetime.now() - timedelta(minutes=SESSION_IDLE_MINUTES):
        conn.execute('DELETE FROM sessions WHERE token_hash = ?', (th,))
        conn.commit()
        return None
    conn.execute('UPDATE sessions SET last_activity = ? WHERE token_hash = ?', (db.now(), th))
    conn.commit()
    return dict(row)


def set_session_cookie(response, token):
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite='Strict',
                        secure=request.is_secure, max_age=12 * 3600)
    return response


# ============================================================================
# YETKİ DEKORATÖRÜ
# ============================================================================

def has_permission(user, permission):
    return bool(user) and user['role'] in PERMISSIONS[permission]


def require(permission):
    """API ucunu oturum ve izin kontrolüyle korur. g.user ve g.db hazırlanır."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            conn = g.db
            user = load_user(conn)
            if not user:
                return jsonify({'success': False, 'error': 'Oturum açmanız gerekiyor', 'code': 'auth_required'}), 401
            if not has_permission(user, permission):
                db.audit(conn, user, 'auth.forbidden', ip=client_ip(), success=False,
                         details={'permission': permission, 'path': request.path})
                conn.commit()
                return jsonify({'success': False, 'code': 'forbidden',
                                'error': f'Bu işlem için yetkiniz yok ({ROLE_LABELS.get(user["role"])})'}), 403
            g.user = user
            return fn(*args, **kwargs)
        return wrapper
    return decorator
