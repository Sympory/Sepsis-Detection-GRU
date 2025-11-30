"""
Authentication and Authorization Module
========================================

Handles user authentication, session management, and role-based access control.
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session, g
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'sepsis_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# Session configuration
SESSION_TIMEOUT_MINUTES = 30
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(**DB_CONFIG)


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def generate_session_id() -> str:
    """Generate secure session ID"""
    return secrets.token_urlsafe(32)


def create_session(user_id: int, ip_address: str, user_agent: str) -> str:
    """Create new user session"""
    session_id = generate_session_id()
    expires_at = datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO sessions (id, user_id, ip_address, user_agent, expires_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, user_id, ip_address, user_agent, expires_at))
        conn.commit()
        return session_id
    finally:
        cur.close()
        conn.close()


def get_session_user(session_id: str):
    """Get user from session ID"""
    if not session_id:
        return None
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Check if session exists and is valid
        cur.execute("""
            SELECT s.*, u.id as user_id, u.username, u.full_name, u.email,
                   u.hospital_id, u.role, u.is_active, h.name as hospital_name,
                   h.code as hospital_code
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            JOIN hospitals h ON u.hospital_id = h.id
            WHERE s.id = %s AND s.expires_at > NOW() AND u.is_active = TRUE
        """, (session_id,))
        
        session_data = cur.fetchone()
        
        if session_data:
            # Update last activity
            cur.execute("""
                UPDATE sessions 
                SET last_activity = NOW(),
                    expires_at = NOW() + INTERVAL '%s minutes'
                WHERE id = %s
            """, (SESSION_TIMEOUT_MINUTES, session_id))
            conn.commit()
            
            return dict(session_data)
        
        return None
    finally:
        cur.close()
        conn.close()


def delete_session(session_id: str):
    """Delete session (logout)"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def delete_user_sessions(user_id: int):
    """Delete all sessions for a user"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def log_audit(user_id, username, hospital_id, action, resource_type=None, 
              resource_id=None, details=None, ip_address=None, user_agent=None,
              success=True, error_message=None):
    """Log audit event"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO audit_logs (
                user_id, username, hospital_id, action, resource_type, resource_id,
                details, ip_address, user_agent, success, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, username, hospital_id, action, resource_type, resource_id,
              details, ip_address, user_agent, success, error_message))
        conn.commit()
    except Exception as e:
        print(f"Audit log error: {e}")
    finally:
        cur.close()
        conn.close()


def authenticate_user(username: str, password: str, ip_address: str, user_agent: str):
    """Authenticate user and create session"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get user
        cur.execute("""
            SELECT u.*, h.name as hospital_name, h.code as hospital_code
            FROM users u
            JOIN hospitals h ON u.hospital_id = h.id
            WHERE u.username = %s
        """, (username,))
        
        user = cur.fetchone()
        
        if not user:
            log_audit(None, username, None, 'LOGIN_FAILED', details='User not found',
                     ip_address=ip_address, user_agent=user_agent, success=False,
                     error_message='Invalid username or password')
            return None, 'Invalid username or password'
        
        user = dict(user)
        
        # Check if account is locked
        if user['locked_until'] and user['locked_until'] > datetime.now():
            log_audit(user['id'], username, user['hospital_id'], 'LOGIN_FAILED',
                     details='Account locked', ip_address=ip_address,
                     user_agent=user_agent, success=False,
                     error_message='Account temporarily locked')
            remaining = (user['locked_until'] - datetime.now()).seconds // 60
            return None, f'Account locked. Try again in {remaining} minutes'
        
        # Check if account is active
        if not user['is_active']:
            log_audit(user['id'], username, user['hospital_id'], 'LOGIN_FAILED',
                     details='Account inactive', ip_address=ip_address,
                     user_agent=user_agent, success=False,
                     error_message='Account inactive')
            return None, 'Account is inactive'
        
        # Verify password
        if not verify_password(password, user['password_hash']):
            # Increment failed attempts
            failed_attempts = user['failed_login_attempts'] + 1
            
            if failed_attempts >= MAX_LOGIN_ATTEMPTS:
                # Lock account
                locked_until = datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                cur.execute("""
                    UPDATE users 
                    SET failed_login_attempts = %s, locked_until = %s
                    WHERE id = %s
                """, (failed_attempts, locked_until, user['id']))
                conn.commit()
                
                log_audit(user['id'], username, user['hospital_id'], 'ACCOUNT_LOCKED',
                         details=f'Too many failed attempts ({failed_attempts})',
                         ip_address=ip_address, user_agent=user_agent)
                
                return None, f'Too many failed attempts. Account locked for {LOCKOUT_DURATION_MINUTES} minutes'
            else:
                cur.execute("""
                    UPDATE users 
                    SET failed_login_attempts = %s
                    WHERE id = %s
                """, (failed_attempts, user['id']))
                conn.commit()
                
                log_audit(user['id'], username, user['hospital_id'], 'LOGIN_FAILED',
                         details=f'Invalid password (attempt {failed_attempts})',
                         ip_address=ip_address, user_agent=user_agent,
                         success=False, error_message='Invalid password')
                
                return None, f'Invalid username or password ({MAX_LOGIN_ATTEMPTS - failed_attempts} attempts remaining)'
        
        # Successful login - reset failed attempts
        cur.execute("""
            UPDATE users 
            SET failed_login_attempts = 0, locked_until = NULL, last_login = NOW()
            WHERE id = %s
        """, (user['id'],))
        conn.commit()
        
        # Create session
        session_id = create_session(user['id'], ip_address, user_agent)
        
        # Log success
        log_audit(user['id'], username, user['hospital_id'], 'LOGIN_SUCCESS',
                 ip_address=ip_address, user_agent=user_agent)
        
        # Return user info (without password hash)
        user_info = {
            'id': user['id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'email': user['email'],
            'hospital_id': user['hospital_id'],
            'hospital_name': user['hospital_name'],
            'hospital_code': user['hospital_code'],
            'role': user['role'],
            'session_id': session_id
        }
        
        return user_info, None
        
    finally:
        cur.close()
        conn.close()


# ============================================================================
# FLASK DECORATORS
# ============================================================================

def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = request.cookies.get('session_id') or request.headers.get('X-Session-ID')
        
        if not session_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        user = get_session_user(session_id)
        
        if not user:
            return jsonify({'error': 'Invalid or expired session'}), 401
        
        # Store user in Flask g object
        g.current_user = user
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(*allowed_roles):
    """Decorator to require specific role(s)"""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if g.current_user['role'] not in allowed_roles:
                log_audit(
                    g.current_user['id'],
                    g.current_user['username'],
                    g.current_user['hospital_id'],
                    'ACCESS_DENIED',
                    details=f"Required roles: {', '.join(allowed_roles)}",
                    ip_address=request.remote_addr,
                    success=False
                )
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def same_hospital_required(f):
    """Decorator to ensure user can only access data from their hospital"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        # Admin can access all hospitals
        if g.current_user['role'] == 'admin':
            return f(*args, **kwargs)
        
        # Check if resource belongs to same hospital
        # This will be implemented per-endpoint basis
        
        return f(*args, **kwargs)
    
    return decorated_function
