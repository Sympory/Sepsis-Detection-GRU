"""
Authentication Endpoints for Flask App
======================================

Add these routes to app.py to enable authentication.
"""

from flask import Flask, request, jsonify, make_response, g
from auth import (
    authenticate_user,
    get_session_user,
    delete_session,
    delete_user_sessions,
    log_audit,
    login_required,
    require_role,
    get_db_connection
)
from psycopg2.extras import RealDictCursor


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/api/hospitals', methods=['GET'])
def get_hospitals():
    """Get list of active hospitals for login dropdown"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, name, code, city
            FROM hospitals
            WHERE is_active = TRUE
            ORDER BY name
        """)
        
        hospitals = [dict(h) for h in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'hospitals': hospitals
        }), 200
        
    except Exception as e:
        print(f"Error fetching hospitals: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load hospitals'
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        username = data.get('username')
        password = data.get('password')
        hospital_id = data.get('hospital_id')
        remember_me = data.get('remember_me', False)
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password required'
            }), 400
        
        # Get client info
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        # Authenticate
        user_info, error = authenticate_user(username, password, ip_address, user_agent)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 401
        
        # Create response
        response = make_response(jsonify({
            'success': True,
            'user': {
                'id': user_info['id'],
                'username': user_info['username'],
                'full_name': user_info['full_name'],
                'email': user_info['email'],
                'hospital_id': user_info['hospital_id'],
                'hospital_name': user_info['hospital_name'],
                'role': user_info['role']
            },
            'session_id': user_info['session_id']
        }))
        
        # Set session cookie
        max_age = 30 * 24 * 60 * 60 if remember_me else None  # 30 days or session
        response.set_cookie(
            'session_id',
            user_info['session_id'],
            max_age=max_age,
            httponly=True,
            samesite='Strict',
            secure=request.is_secure
        )
        
        return response, 200
        
    except Exception as e:
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Login failed'
        }), 500


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """User logout endpoint"""
    try:
        session_id = request.cookies.get('session_id') or request.headers.get('X-Session-ID')
        
        if session_id:
            # Log audit
            log_audit(
                g.current_user['id'],
                g.current_user['username'],
                g.current_user['hospital_id'],
                'LOGOUT_SUCCESS',
                ip_address=request.remote_addr
            )
            
            # Delete session
            delete_session(session_id)
        
        # Create response
        response = make_response(jsonify({
            'success': True,
            'message': 'Logged out successfully'
        }))
        
        # Clear session cookie
        response.set_cookie('session_id', '', max_age=0)
        
        return response, 200
        
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({
            'success': False,
            'error': 'Logout failed'
        }), 500


@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current logged-in user info"""
    try:
        user = g.current_user
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['user_id'],
                'username': user['username'],
                'full_name': user['full_name'],
                'email': user['email'],
                'hospital_id': user['hospital_id'],
                'hospital_name': user['hospital_name'],
                'hospital_code': user['hospital_code'],
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        print(f"Get current user error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get user info'
        }), 500


@app.route('/api/auth/change-password', methods=['POST'])
@login_required
def change_password():
    """Change password for current user"""
    try:
        data = request.get_json()
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({
                'success': False,
                'error': 'Current and new passwords required'
            }), 400
        
        if len(new_password) < 8:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 8 characters'
            }), 400
        
        # Verify current password
        from auth import verify_password, hash_password
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT password_hash FROM users WHERE id = %s", (g.current_user['user_id'],))
        user = cur.fetchone()
        
        if not user or not verify_password(current_password, user['password_hash']):
            cur.close()
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Current password is incorrect'
            }), 401
        
        # Update password
        new_hash = hash_password(new_password)
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", 
                   (new_hash, g.current_user['user_id']))
        conn.commit()
        
        # Delete all other sessions
        delete_user_sessions(g.current_user['user_id'])
        
        # Log audit
        log_audit(
            g.current_user['user_id'],
            g.current_user['username'],
            g.current_user['hospital_id'],
            'PASSWORD_CHANGED',
            ip_address=request.remote_addr
        )
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        print(f"Change password error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to change password'
        }), 500


# ============================================================================
# USER MANAGEMENT ENDPOINTS (Admin only)
# ============================================================================

@app.route('/api/users', methods=['GET'])
@require_role('admin', 'hospital_admin')
def get_users():
    """Get users list (filtered by hospital for hospital_admin)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Filter by hospital for hospital_admin
        if g.current_user['role'] == 'hospital_admin':
            cur.execute("""
                SELECT u.id, u.username, u.full_name, u.email, u.role, 
                       u.is_active, u.last_login, u.created_at,
                       h.name as hospital_name, h.code as hospital_code
                FROM users u
                JOIN hospitals h ON u.hospital_id = h.id
                WHERE u.hospital_id = %s
                ORDER BY u.created_at DESC
            """, (g.current_user['hospital_id'],))
        else:
            cur.execute("""
                SELECT u.id, u.username, u.full_name, u.email, u.role, 
                       u.is_active, u.last_login, u.created_at,
                       h.name as hospital_name, h.code as hospital_code
                FROM users u
                JOIN hospitals h ON u.hospital_id = h.id
                ORDER BY u.created_at DESC
            """)
        
        users = [dict(u) for u in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'users': users
        }), 200
        
    except Exception as e:
        print(f"Get users error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get users'
        }), 500


@app.route('/api/audit-logs', methods=['GET'])
@require_role('admin', 'hospital_admin')
def get_audit_logs():
    """Get audit logs (filtered by hospital for hospital_admin)"""
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Filter by hospital for hospital_admin
        if g.current_user['role'] == 'hospital_admin':
            cur.execute("""
                SELECT * FROM v_recent_audit_activity
                WHERE hospital_name = (
                    SELECT name FROM hospitals WHERE id = %s
                )
                LIMIT %s OFFSET %s
            """, (g.current_user['hospital_id'], limit, offset))
        else:
            cur.execute("""
                SELECT * FROM v_recent_audit_activity
                LIMIT %s OFFSET %s
            """, (limit, offset))
        
        logs = [dict(log) for log in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'logs': logs,
            'count': len(logs)
        }), 200
        
    except Exception as e:
        print(f"Get audit logs error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get audit logs'
        }), 500


# ============================================================================
# UPDATE EXISTING PATIENT ENDPOINTS TO REQUIRE AUTHENTICATION
# ============================================================================

# Note: Update existing patient endpoints in app.py with @login_required decorator
# Example:
#
# @app.route('/api/patients', methods=['GET'])
# @login_required
# def get_patients():
#     # Filter by hospital
#     if g.current_user['role'] != 'admin':
#         # Only show patients from same hospital
#         hospital_filter = g.current_user['hospital_id']
#     ...
