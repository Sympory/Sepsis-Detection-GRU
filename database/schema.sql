-- ============================================================================
-- Database Schema for Multi-User Authentication System
-- ============================================================================
-- PostgreSQL schema with hospitals, users, sessions, and audit logging.
-- ============================================================================

-- ============================================================================
-- HOSPITALS TABLE
-- ============================================================================
CREATE TABLE hospitals (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100) DEFAULT 'Turkey',
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hospitals_code ON hospitals(code);
CREATE INDEX idx_hospitals_active ON hospitals(is_active);

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    hospital_id INTEGER NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'hospital_admin', 'doctor', 'nurse', 'viewer')),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id)
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_hospital ON users(hospital_id);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active);

-- ============================================================================
-- SESSIONS TABLE
-- ============================================================================
CREATE TABLE sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);

-- ============================================================================
-- AUDIT LOGS TABLE
-- ============================================================================
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(100),
    hospital_id INTEGER REFERENCES hospitals(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id INTEGER,
    details TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_hospital ON audit_logs(hospital_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- PATIENTS TABLE (Updated with hospital_id)
-- ============================================================================
DROP TABLE IF EXISTS patients CASCADE;

CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    hospital_id INTEGER NOT NULL REFERENCES hospitals(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    admission_time TIMESTAMP,
    discharge_time TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(patient_id, hospital_id)
);

CREATE INDEX idx_patients_hospital ON patients(hospital_id);
CREATE INDEX idx_patients_patient_id ON patients(patient_id);
CREATE INDEX idx_patients_active ON patients(is_active);
CREATE INDEX idx_patients_admission ON patients(admission_time);

-- ============================================================================
-- HOURLY DATA TABLE (Updated)
-- ============================================================================
DROP TABLE IF EXISTS hourly_data CASCADE;

CREATE TABLE hourly_data (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    hour INTEGER NOT NULL,
    vital_signs JSONB NOT NULL,
    prediction REAL,
    risk_level VARCHAR(50),
    threshold_exceeded BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    entered_by INTEGER REFERENCES users(id),
    UNIQUE(patient_id, hour)
);

CREATE INDEX idx_hourly_patient ON hourly_data(patient_id);
CREATE INDEX idx_hourly_hour ON hourly_data(hour);
CREATE INDEX idx_hourly_prediction ON hourly_data(prediction);
CREATE INDEX idx_hourly_risk ON hourly_data(risk_level);
CREATE INDEX idx_hourly_timestamp ON hourly_data(timestamp);

-- ============================================================================
-- TRIGGERS FOR updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_hospitals_updated_at BEFORE UPDATE ON hospitals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active users by hospital
CREATE VIEW v_active_users_by_hospital AS
SELECT 
    h.id as hospital_id,
    h.name as hospital_name,
    h.code as hospital_code,
    COUNT(u.id) as total_users,
    COUNT(CASE WHEN u.role = 'doctor' THEN 1 END) as doctors,
    COUNT(CASE WHEN u.role = 'nurse' THEN 1 END) as nurses,
    COUNT(CASE WHEN u.role = 'hospital_admin' THEN 1 END) as admins
FROM hospitals h
LEFT JOIN users u ON h.id = u.hospital_id AND u.is_active = TRUE
GROUP BY h.id, h.name, h.code;

-- Active patients by hospital
CREATE VIEW v_active_patients_by_hospital AS
SELECT 
    h.id as hospital_id,
    h.name as hospital_name,
    COUNT(p.id) as active_patients,
    COUNT(CASE WHEN h2.prediction > 0.5 THEN 1 END) as high_risk_patients
FROM hospitals h
LEFT JOIN patients p ON h.id = p.hospital_id AND p.is_active = TRUE
LEFT JOIN LATERAL (
    SELECT prediction 
    FROM hourly_data 
    WHERE patient_id = p.id 
    ORDER BY hour DESC 
    LIMIT 1
) h2 ON TRUE
GROUP BY h.id, h.name;

-- Recent audit activity
CREATE VIEW v_recent_audit_activity AS
SELECT 
    a.id,
    a.username,
    h.name as hospital_name,
    a.action,
    a.resource_type,
    a.timestamp,
    a.success
FROM audit_logs a
LEFT JOIN hospitals h ON a.hospital_id = h.id
ORDER BY a.timestamp DESC
LIMIT 100;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE hospitals IS 'Hospital/organization information for multi-tenancy';
COMMENT ON TABLE users IS 'System users with role-based access control';
COMMENT ON TABLE sessions IS 'Active user sessions for authentication';
COMMENT ON TABLE audit_logs IS 'Complete audit trail of all system actions';
COMMENT ON TABLE patients IS 'Patient records, linked to specific hospitals';
COMMENT ON TABLE hourly_data IS 'Hourly vital signs and predictions for patients';
