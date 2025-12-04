"""
Sepsis Tahmin Sistemi - Hasta Takip Web Arayüzü
================================================

Flask web sunucusu ile GRU modelini yükler ve hasta takip API'si sağlar.

Endpoints:
    GET  /                    : Ana sayfa (hasta listesi)
    GET  /api/patients        : Tüm hastaları listele
    POST /api/patients        : Yeni hasta ekle
    GET  /api/patients/<id>   : Hasta detaylarını getir
    POST /api/patients/<id>/hourly-data : Saatlik veri ekle ve tahmin yap
    DELETE /api/patients/<id> : Hasta sil
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import os
import traceback
import sqlite3
import json
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

# Global değişkenler
model = None

# Dosya yolları
DB_PATH = 'patients.db'
MODEL_PATH = 'models/gru_v23_best.keras'
PREPROCESSING_DIR = 'data/processed'

def init_database():
    """Veritabanını oluştur ve tabloları tanımla"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Hastalar tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            admission_time TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Saatlik veriler tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            hour INTEGER NOT NULL,
            vital_signs TEXT NOT NULL,
            prediction REAL,
            risk_level TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            UNIQUE(patient_id, hour)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Veritabanı başlatıldı")


def get_db():
    """Veritabanı bağlantısı al"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# VALIDATION RANGES & FUNCTIONS
# ============================================================================

VITAL_SIGN_RANGES = {
    # Vital Signs
    'HR': (40, 200),
    'Temp': (35, 42),
    'SBP': (60, 250),
    'DBP': (30, 150),
    'MAP': (40, 180),
    'Resp': (8, 50),
    'O2Sat': (70, 100),
    'EtCO2': (10, 80),
    
    # Lab - Hematology
    'WBC': (1, 50),
    'Platelets': (20, 800),
    'Hgb': (5, 20),
    'Hct': (15, 65),
    
    # Lab - Chemistry  
    'Creatinine': (0.3, 15),
    'BUN': (3, 150),
    'Glucose': (30, 600),
    'Lactate': (0.5, 20),
    'Bilirubin_total': (0.1, 30),
    'Bilirubin_direct': (0, 15),
    
    # Lab - ABG
    'pH': (6.8, 7.8),
    'PaCO2': (15, 100),
    'PaO2': (40, 500),
    'HCO3': (10, 45),
    'BaseExcess': (-20, 20),
    
    # Lab - Electrolytes
    'Calcium': (5, 15),
    'Chloride': (70, 130),
    'Potassium': (2, 8),
    'Magnesium': (0.5, 5),
    
    # Lab - Liver
    'AST': (5, 5000),
    'ALT': (5, 5000),
    'ALP': (20, 1000),
    
    # Biomarkers
    'PCT': (0.01, 100),
    'CRP': (0, 500),
    'Presepsin': (100, 5000),
    'IL6': (0, 1000),
    'IL1b': (0, 200),
    'ESR': (0, 150),
    'MDW': (15, 40),
    'MPV': (5, 15),
    'RDW': (10, 25),
    'Neutrophils': (0.5, 40),
    'Lymphocytes': (0.2, 10),
    'DDimer': (0, 20),
    'PT': (8, 50),
    'aPTT': (15, 100),
    'INR': (0.5, 10),
    'IonizedCalcium': (0.8, 1.5),
    'Phosphorus': (1, 10),
    'Albumin': (1.5, 6),
    'Sodium': (110, 170),
    'NLR': (0.1, 100),
    'PLR': (10, 1000),
    'AnionGap': (0, 40),
    'Urine_output': (0, 500),
}

@app.route('/')
def index():
    """Ana sayfa"""
    return send_from_directory('.', 'index.html')

@app.route('/login')
def login_page():
    """Giriş sayfası"""
    return send_from_directory('.', 'login.html')

def validate_vital_signs(vital_signs):
    """
    Validate vital signs are within acceptable ranges
    Returns: (is_valid, errors_list)
    """
    errors = []
    
    for field, value in vital_signs.items():
        # Skip empty values (let imputer handle them)
        if value is None or value == '':
            continue
            
        if field in VITAL_SIGN_RANGES:
            try:
                num_value = float(value)
                min_val, max_val = VITAL_SIGN_RANGES[field]
                
                if not (min_val <= num_value <= max_val):
                    errors.append(f"{field} must be between {min_val} and {max_val} (got {num_value})")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")
    
    return (len(errors) == 0, errors)


# ============================================================================
# MODEL FONKSİYONLARI
# ============================================================================

def load_model_and_preprocessing():
    """Modeli ve preprocessing objelerini yükle"""
    global model, imputer, scaler, ohe, numerical_columns, categorical_columns
    
    try:
        print("\n" + "="*60)
        print("MODEL VE PREPROCESSING YÜKLENİYOR...")
        print("="*60)
        
        # Modeli yükle
        print(f"\n[1/5] Model yükleniyor: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        print("  ✓ Model başarıyla yüklendi")
        
        # Column info yükle
        print(f"\n[2/5] Sütun bilgileri yükleniyor...")
        with open(os.path.join(PREPROCESSING_DIR, 'column_info.pkl'), 'rb') as f:
            column_info = pickle.load(f)
        numerical_columns = column_info['numerical_columns']
        categorical_columns = column_info.get('categorical_columns', [])
        print(f"  ✓ {len(numerical_columns)} sayısal özellik")
        print(f"  ✓ {len(categorical_columns)} kategorik özellik")
        
        # Imputer yükle
        print(f"\n[3/5] Imputer yükleniyor...")
        with open(os.path.join(PREPROCESSING_DIR, 'imputer.pkl'), 'rb') as f:
            imputer = pickle.load(f)
        print("  ✓ Imputer yüklendi")
        
        # Scaler yükle
        print(f"\n[4/5] Scaler yükleniyor...")
        with open(os.path.join(PREPROCESSING_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("  ✓ Scaler yüklendi")
        
        # OneHotEncoder yükle (varsa)
        print(f"\n[5/5] OneHotEncoder kontrol ediliyor...")
        ohe_path = os.path.join(PREPROCESSING_DIR, 'ohe.pkl')
        if os.path.exists(ohe_path):
            with open(ohe_path, 'rb') as f:
                ohe = pickle.load(f)
            print("  ✓ OneHotEncoder yüklendi")
        else:
            print("  - OneHotEncoder bulunamadı (opsiyonel)")
        
        print("\n" + "="*60)
        print("✓ TÜM BİLEŞENLER BAŞARIYLA YÜKLENDİ!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ HATA: Model yüklenemedi!")
        print(f"Hata detayı: {str(e)}")
        traceback.print_exc()
        return False


def predict_with_history(hourly_data_list, window_size=6):
    """
    Saatlik veri geçmişine göre kademeli tahmin yap
    
    Args:
        hourly_data_list: List of dicts, her saat için vital signs
                         [hour1_data, hour2_data, ...]
        window_size: Model pencere boyutu (default: 6)
    
    Returns:
        prediction: Risk skoru (0-1)
    """
    num_hours = len(hourly_data_list)
    
    # Yeterli veri yoksa padding ekle (NaN ile)
    if num_hours < window_size:
        # İlk saatleri NaN ile doldur
        padding = [{}] * (window_size - num_hours)
        full_data = padding + hourly_data_list
    else:
        # Son 6 saati al
        full_data = hourly_data_list[-window_size:]
    
    # DataFrame'e dönüştür
    df = pd.DataFrame(full_data)
    
    # Eksik olan sayısal sütunları NaN ile doldur
    for col in numerical_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Eksik olan kategorik sütunları boş string ile doldur
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Sütun sırasını düzenle
    df = df[numerical_columns + categorical_columns]
    
    # Sayısal özellikleri işle
    X_numerical = imputer.transform(df[numerical_columns])
    X_numerical = scaler.transform(X_numerical)
    
    # Kategorik özellikleri işle (varsa)
    if categorical_columns and ohe is not None:
        X_categorical = ohe.transform(df[categorical_columns])
        X_combined = np.hstack([X_numerical, X_categorical])
    else:
        X_combined = X_numerical
    
    # Sekans formatına dönüştür: (1, window_size, num_features)
    X_seq = X_combined.reshape(1, window_size, X_combined.shape[1])
    
    # Tahmin yap
    prediction = model.predict(X_seq, verbose=0)[0][0]
    
    return float(prediction)


def get_risk_level(prediction):
    """Risk seviyesini ve rengini belirle"""
    if prediction < 0.1:
        return "Çok Düşük", "#10b981"  # Green
    elif prediction < 0.3:
        return "Düşük", "#3b82f6"  # Blue
    elif prediction < 0.5:
        return "Orta", "#f59e0b"  # Orange
    elif prediction < 0.7:
        return "Yüksek", "#ef4444"  # Red
    else:
        return "Çok Yüksek", "#dc2626"  # Dark Red


# ============================================================================
# WEB ENDPOINTS
# ============================================================================




@app.route('/<path:path>')
def serve_static(path):
    """Statik dosyaları servis et"""
    return send_from_directory('.', path)


# ============================================================================
# HASTA YÖNETİMİ API
# ============================================================================

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Tüm hastaları listele"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        patients = cursor.execute('''
            SELECT p.*, 
                   COUNT(h.id) as total_hours,
                   MAX(h.prediction) as latest_prediction,
                   MAX(h.risk_level) as latest_risk_level
            FROM patients p
            LEFT JOIN hourly_data h ON p.id = h.patient_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        ''').fetchall()
        
        conn.close()
        
        patients_list = [{
            'id': p['id'],
            'patient_id': p['patient_id'],
            'name': p['name'],
            'age': p['age'],
            'gender': p['gender'],
            'admission_time': p['admission_time'],
            'total_hours': p['total_hours'],
            'latest_prediction': p['latest_prediction'],
            'latest_risk_level': p['latest_risk_level'],
            'created_at': p['created_at']
        } for p in patients]
        
        return jsonify({
            'success': True,
            'patients': patients_list
        }), 200
        
    except Exception as e:
        print(f"❌ Hasta listesi hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patients', methods=['POST'])
def create_patient():
    """Yeni hasta ekle"""
    try:
        data = request.get_json()
        
        required_fields = ['patient_id', 'name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Eksik alan: {field}'
                }), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (patient_id, name, age, gender, admission_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data['patient_id'],
            data['name'],
            data.get('age'),
            data.get('gender'),
            data.get('admission_time', datetime.now().isoformat())
        ))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✓ Yeni hasta eklendi: {data['name']} (ID: {patient_id})")
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'message': 'Hasta başarıyla eklendi'
        }), 201
        
    except sqlite3.IntegrityError:
        return jsonify({
            'success': False,
            'error': 'Bu hasta ID zaten mevcut'
        }), 400
    except Exception as e:
        print(f"❌ Hasta ekleme hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Hasta detaylarını ve tüm saatlik verileri getir"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Hasta bilgisi
        patient = cursor.execute(
            'SELECT * FROM patients WHERE id = ?',
            (patient_id,)
        ).fetchone()
        
        if not patient:
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Hasta bulunamadı'
            }), 404
        
        # Saatlik veriler
        hourly_data = cursor.execute('''
            SELECT * FROM hourly_data 
            WHERE patient_id = ? 
            ORDER BY hour ASC
        ''', (patient_id,)).fetchall()
        
        conn.close()
        
        hourly_list = [{
            'id': h['id'],
            'hour': h['hour'],
            'vital_signs': json.loads(h['vital_signs']),
            'prediction': h['prediction'],
            'risk_level': h['risk_level'],
            'timestamp': h['timestamp']
        } for h in hourly_data]
        
        return jsonify({
            'success': True,
            'patient': {
                'id': patient['id'],
                'patient_id': patient['patient_id'],
                'name': patient['name'],
                'age': patient['age'],
                'gender': patient['gender'],
                'admission_time': patient['admission_time'],
                'created_at': patient['created_at']
            },
            'hourly_data': hourly_list
        }), 200
        
    except Exception as e:
        print(f"❌ Hasta detay hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patients/<int:patient_id>/hourly-data', methods=['POST'])
def add_hourly_data(patient_id):
    """
    Hastaya saatlik veri ekle ve kademeli tahmin yap
    """
    try:
        data = request.get_json()
        
        if 'hour' not in data or 'vital_signs' not in data:
            return jsonify({
                'success': False,
                'error': 'hour ve vital_signs alanları gerekli'
            }), 400
        
        hour = data['hour']
        vital_signs = data['vital_signs']
        
        # VALIDATION - Validate vital signs
        is_valid, errors = validate_vital_signs(vital_signs)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': errors
            }), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Hasta kontrolü
        patient = cursor.execute(
            'SELECT * FROM patients WHERE id = ?',
            (patient_id,)
        ).fetchone()
        
        if not patient:
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Hasta bulunamadı'
            }), 404
        
        # Önceki saatleri al
        previous_hours = cursor.execute('''
            SELECT vital_signs FROM hourly_data 
            WHERE patient_id = ? AND hour < ?
            ORDER BY hour ASC
        ''', (patient_id, hour)).fetchall()
        
        # Saatlik veri listesi oluştur
        hourly_data_list = [json.loads(h['vital_signs']) for h in previous_hours]
        hourly_data_list.append(vital_signs)
        
        # Kademeli tahmin yap
        prediction = predict_with_history(hourly_data_list)
        risk_level, risk_color = get_risk_level(prediction)
        
        # Veritabanına kaydet
        cursor.execute('''
            INSERT OR REPLACE INTO hourly_data 
            (patient_id, hour, vital_signs, prediction, risk_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            patient_id,
            hour,
            json.dumps(vital_signs),
            prediction,
            risk_level
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Saat {hour} verisi eklendi: {patient['name']} - Risk: {prediction:.4f} ({risk_level})")
        
        return jsonify({
            'success': True,
            'hour': hour,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'is_sepsis_risk': prediction >= 0.1799,
            'message': f'Saat {hour} verisi kaydedildi ve tahmin yapıldı'
        }), 201
        
    except Exception as e:
        print(f"❌ Saatlik veri ekleme hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/patients/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Hasta sil"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Hasta kontrolü
        patient = cursor.execute(
            'SELECT name FROM patients WHERE id = ?',
            (patient_id,)
        ).fetchone()
        
        if not patient:
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Hasta bulunamadı'
            }), 404
        
        # Saatlik verileri sil
        cursor.execute('DELETE FROM hourly_data WHERE patient_id = ?', (patient_id,))
        
        # Hastayı sil
        cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Hasta silindi: {patient['name']} (ID: {patient_id})")
        
        return jsonify({
            'success': True,
            'message': 'Hasta başarıyla silindi'
        }), 200
        
    except Exception as e:
        print(f"❌ Hasta silme hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# AUTHENTICATION ENDPOINTS (Demo Mode for SQLite)
# ============================================================================

@app.route('/api/hospitals', methods=['GET'])
def get_hospitals():
    """Get list of hospitals for login"""
    return jsonify({
        'success': True,
        'hospitals': [{'id': 1, 'name': 'Demo Hospital', 'code': 'DEMO'}]
    }), 200


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Demo login - accepts any credentials"""
    data = request.get_json() or {}
    return jsonify({
        'success': True,
        'user': {
            'id': 1,
            'username': data.get('username', 'demo'),
            'role': 'doctor',
            'hospital_name': 'Demo Hospital'
        }
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout"""
    return jsonify({'success': True}), 200


@app.route('/api/auth/me', methods=['GET'])
def get_current_user():
    """Get current user info"""
    return jsonify({
        'success': True,
        'user': {'id': 1, 'username': 'demo', 'role': 'doctor'}
    }), 200


@app.route('/api/health', methods=['GET'])
def health():
    """Sağlık kontrolü endpoint'i"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessing_loaded': all([imputer is not None, scaler is not None]),
        'database_exists': os.path.exists(DB_PATH)
    }), 200


# ============================================================================
# SERVER BAŞLATMA
# ============================================================================

if __name__ == '__main__':
    # Veritabanını başlat
    init_database()
    
    # Modeli yükle
    if load_model_and_preprocessing():
        print("\n🚀 Flask sunucusu başlatılıyor...\n")
        print("="*60)
        print("🌐 Web arayüzüne erişim: http://localhost:5000")
        print("📊 Hasta yönetim API'si hazır")
        print("⌨️  Çıkmak için: CTRL+C")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Sunucu başlatılamadı. Model yükleme hatası!")
