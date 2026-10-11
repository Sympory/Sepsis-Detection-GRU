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
# --- BU KISMI 'import' SATIRLARININ HEMEN ALTINA YAPIŞTIR ---

from tensorflow.keras import layers
import tensorflow.keras.backend as K

# --- GÜNCEL KOD BAŞLANGICI (Optimized Model ile Uyumlu) ---

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Binary form of focal loss.
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(loss, axis=1)
    return binary_focal_loss_fixed

class AttentionLayer(layers.Layer):
    """Temporal Attention Mechanism"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.time_steps = input_shape[1]

        if self.time_steps is None:
             pass

        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal',
                                 trainable=True)

        if self.time_steps is not None:
            self.b = self.add_weight(name='attention_bias',
                                     shape=(self.time_steps, 1),
                                     initializer='zeros',
                                     trainable=True)
        else:
            self.b = None

        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.dot(x, self.W)
        if self.b is not None:
            e += self.b    
        e = K.tanh(e)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# --- GÜNCEL KOD BİTİŞİ ---

app = Flask(__name__, static_folder='.')
CORS(app)

# Global değişkenler
model = None
imputer = None
scaler = None
ohe = None
transformer = None
feature_stats = {}
estimator = None
numerical_columns = []
categorical_columns = []

WINDOW_SIZE = 6

# Dosya yolları - Merkezi config'den oku
from config import (DB_PATH, MODEL_PATH, PREPROCESSING_DIR, OPTIMAL_THRESHOLD, MODEL_INFO,
                    RISK_LEVELS, CLINICAL_OVERRIDES, FEATURE_SET, HOST, PORT)
from features import build_hourly_features, feature_names, make_windows
from reliability import (ReliabilityEstimator, apply_clinical_disagreement, mc_dropout_predict,
                         normalize_inputs)

# Modelin girdisinde kullanılmayan (eğitimde hariç tutulan) alanlar — yalnızca kayıt amaçlı
MODEL_UNUSED_FIELDS = ['PCT', 'CRP', 'Presepsin', 'IL6']

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

    # İndeksler (sorgu performansı için)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_patient ON hourly_data(patient_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_hour ON hourly_data(patient_id, hour)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_patients_pid ON patients(patient_id)')

    # Eski veritabanları için güvenilirlik sütunları (migration)
    existing = {row[1] for row in cursor.execute('PRAGMA table_info(hourly_data)')}
    if 'reliability' not in existing:
        cursor.execute('ALTER TABLE hourly_data ADD COLUMN reliability REAL')
    if 'assessment' not in existing:
        cursor.execute('ALTER TABLE hourly_data ADD COLUMN assessment TEXT')

    conn.commit()
    conn.close()
    print("[OK] Veritabani baslatildi (indeksler olusturuldu)")


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

    # Formda olup yukarıda eksik kalanlar
    'FiO2': (0.21, 100),          # 0.21-1.0 oran veya 21-100 (%) kabul edilir
    'SaO2': (50, 100),
    'Alkalinephos': (10, 2000),
    'Phosphate': (0.5, 15),
    'PTT': (15, 150),
    'Fibrinogen': (30, 1200),
    'TroponinI': (0, 100),
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
    global model, estimator

    try:
        print("\n" + "="*60)
        print("MODEL VE PREPROCESSING YUKLENIYOR...")
        print("="*60)

        # Modeli yukle
        print(f"\n[1/2] Model yukleniyor ({FEATURE_SET}): {MODEL_PATH}")

        # Custom objects tanimla (tum modeller icin)
        custom_objects = {
            'binary_focal_loss': binary_focal_loss,
            'binary_focal_loss_fixed': binary_focal_loss(gamma=2.0, alpha=0.25),
            'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25),
            'AttentionLayer': AttentionLayer,
        }

        try:
            model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            print("  [OK] Model basariyla yuklendi")
        except Exception as e1:
            # Standart yukleme dene
            try:
                model = keras.models.load_model(MODEL_PATH)
                print("  [OK] Model (standart) basariyla yuklendi")
            except Exception as e2:
                print(f"  [FAIL] Model yukleme hatasi: {str(e2)}")
                raise e2

        if FEATURE_SET == 'v3':
            load_v3_preprocessing()
        else:
            load_legacy_preprocessing()

        # Güvenilirlik tahmincisi (kalibratör varsa yüklenir)
        estimator = ReliabilityEstimator(feature_stats, OPTIMAL_THRESHOLD,
                                         calibration_dir=os.path.dirname(MODEL_PATH))
        if estimator.calibrator is None:
            print("  [WARN] Kalibrator yok - 'python scripts/fit_calibration.py' calistirin")
        else:
            print("  [OK] Kalibrator ve guvenilirlik istatistikleri yuklendi")

        print("\n" + "="*60)
        print("[OK] TUM BILESENLER BASARIYLA YUKLENDI!")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n[FAIL] HATA: Model yuklenemedi!")
        print(f"Hata detayi: {str(e)}")
        traceback.print_exc()
        return False


def load_v3_preprocessing():
    """v3: ileri doldurma + ölçüm maskeleri (features.py), eğitim hastalarıyla fit edilmiş dönüştürücü"""
    global transformer, feature_stats
    print("\n[2/2] v3 ozellik donusturucu yukleniyor...")
    with open(os.path.join(os.path.dirname(MODEL_PATH), 'transformer.pkl'), 'rb') as f:
        transformer = pickle.load(f)
    feature_stats = transformer.clinical_stats()
    print(f"  [OK] {len(transformer.names)} ozellik (34 deger + 34 olcum maskesi + statik)")


def load_legacy_preprocessing():
    """Eski model: SimpleImputer(median) + StandardScaler, data/processed altındaki nesneler"""
    global imputer, scaler, ohe, numerical_columns, categorical_columns, feature_stats
    print("\n[2/2] Eski preprocessing nesneleri yukleniyor...")
    with open(os.path.join(PREPROCESSING_DIR, 'column_info.pkl'), 'rb') as f:
        column_info = pickle.load(f)
    numerical_columns = column_info['numerical_columns']
    categorical_columns = column_info.get('categorical_columns', [])
    with open(os.path.join(PREPROCESSING_DIR, 'imputer.pkl'), 'rb') as f:
        imputer = pickle.load(f)
    with open(os.path.join(PREPROCESSING_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    ohe_path = os.path.join(PREPROCESSING_DIR, 'ohe.pkl')
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
    feature_stats = {c: (float(m), float(sd)) for c, m, sd in zip(numerical_columns, scaler.mean_, scaler.scale_)}
    print(f"  [OK] {len(numerical_columns)} sayisal ozellik")


def build_sequence(hourly_data_list, window_size=WINDOW_SIZE):
    """
    Saatlik kayıtlardan (1, window, features) model girdisini oluşturur.

    6 saatten az veri varsa ilk ölçüm geçmişe kopyalanır (backfilling);
    sıfır/medyan ile doldurmak hastayı 'sağlıklı' gösterirdi.
    Dönüş: (X_seq, gerçek saat sayısı, penceredeki gerçek kayıtlar)
    """
    real_rows = hourly_data_list[-window_size:]
    real_hours = len(real_rows)

    if real_hours == 0:
        raise ValueError('Tahmin için en az bir saatlik veri gerekli')

    if FEATURE_SET == 'v3':
        # Tüm geçmiş kullanılır: önceki saatlerde ölçülen lab değerleri ileri taşınır
        feats = build_hourly_features(pd.DataFrame(hourly_data_list))
        X_seq = make_windows(transformer.transform(feats), window_size)[-1:]
    else:
        X_seq = build_legacy_sequence(real_rows, window_size)

    expected = model.input_shape[-1]
    if X_seq.shape[-1] != expected:
        raise ValueError(f'Özellik sayısı uyumsuz: {X_seq.shape[-1]} != model {expected}')

    return X_seq.astype(np.float32), real_hours, real_rows


def build_legacy_sequence(real_rows, window_size):
    padding = [real_rows[0]] * (window_size - len(real_rows))
    df = pd.DataFrame(padding + real_rows)

    for col in numerical_columns:
        if col not in df.columns: df[col] = np.nan
    for col in categorical_columns:
        if col not in df.columns: df[col] = ''

    df = df[numerical_columns + categorical_columns]
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

    X_numerical = scaler.transform(imputer.transform(df[numerical_columns]))

    if categorical_columns and ohe is not None:
        X_combined = np.hstack([X_numerical, ohe.transform(df[categorical_columns])])
    else:
        X_combined = X_numerical

    return X_combined.reshape(1, window_size, -1)


def predict_with_history(hourly_data_list, window_size=WINDOW_SIZE):
    """Deterministik tahmin + belirsizlik için MC-Dropout örnekleri"""
    X_seq, real_hours, real_rows = build_sequence(hourly_data_list, window_size)
    prediction = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
    mc_samples = mc_dropout_predict(model, X_seq)
    return prediction, mc_samples, real_hours, real_rows


def get_risk_level(prediction, vital_signs_dict=None):
    """
    Risk seviyesini belirler: (etiket, renk, tetiklenen klinik bayraklar).

    Klinik Override: Model düşük tahmin verse bile kritik vital bulgular
    varsa risk yükseltilir. Kurallar config.CLINICAL_OVERRIDES'tadır.
    """
    vs = vital_signs_dict or {}
    flags = []
    for name, rule in CLINICAL_OVERRIDES.items():
        feature = name.rsplit('_', 1)[0]
        value = vs.get(feature)
        if value is None or value == '':
            continue
        value = float(value)
        hit = value < rule['threshold'] if rule['direction'] == 'below' else value > rule['threshold']
        if hit:
            flags.append({'feature': feature, 'value': value, 'rule': name, 'points': rule['red_flags']})

    points = sum(f['points'] for f in flags)
    adjusted_prediction = prediction
    if points >= 5:
        adjusted_prediction = max(prediction, 0.60)
    elif points >= 3:
        adjusted_prediction = max(prediction, 0.35)
    elif points >= 2:
        adjusted_prediction = max(prediction, 0.25)

    for key in ('critical', 'high', 'moderate', 'low', 'very_low'):
        level = RISK_LEVELS[key]
        if adjusted_prediction >= level['min']:
            return level['label'], level['color'], flags
    return RISK_LEVELS['very_low']['label'], RISK_LEVELS['very_low']['color'], flags


# ============================================================================
# WEB ENDPOINTS
# ============================================================================




# Yalnızca arayüz dosyaları servis edilir; proje klasöründeki diğer dosyalar
# (patients.db, .env, modeller) dışarı açılmaz.
STATIC_FILES = {'index.html', 'style.css', 'script.js', 'login.html', 'login.js', 'login_style.css'}


@app.route('/<path:path>')
def serve_static(path):
    """Statik dosyaları servis et"""
    if path not in STATIC_FILES:
        return jsonify({'success': False, 'error': 'Bulunamadı'}), 404
    return send_from_directory('.', path)


# ============================================================================
# HASTA YÖNETİMİ API
# ============================================================================

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Tüm hastaları en son saatlik kayıtlarıyla birlikte listele"""
    try:
        conn = get_db()
        patients = conn.execute('''
            SELECT p.*,
                   (SELECT COUNT(*) FROM hourly_data h WHERE h.patient_id = p.id) AS total_hours,
                   l.hour AS latest_hour,
                   l.prediction AS latest_prediction,
                   l.risk_level AS latest_risk_level,
                   l.reliability AS latest_reliability,
                   l.timestamp AS latest_timestamp
            FROM patients p
            LEFT JOIN hourly_data l ON l.id = (
                SELECT id FROM hourly_data WHERE patient_id = p.id ORDER BY hour DESC LIMIT 1
            )
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
            'hourly_data_count': p['total_hours'],
            'latest_hour': p['latest_hour'],
            'latest_prediction': p['latest_prediction'],
            'latest_risk': p['latest_risk_level'],
            'latest_risk_level': p['latest_risk_level'],
            'latest_reliability': p['latest_reliability'],
            'latest_timestamp': p['latest_timestamp'],
            'created_at': p['created_at']
        } for p in patients]

        return jsonify({'success': True, 'patients': patients_list}), 200

    except Exception as e:
        print(f"[FAIL] Hasta listesi hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


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

        print(f"[OK] Yeni hasta eklendi: {data['name']} (ID: {patient_id})")

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
        print(f"[FAIL] Hasta ekleme hatası: {str(e)}")
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
            'timestamp': h['timestamp'],
            'reliability': h['reliability'],
            'assessment': json.loads(h['assessment']) if h['assessment'] else None
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
        print(f"[FAIL] Hasta detay hatası: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/patients/<int:patient_id>/hourly-data', methods=['POST'])
def add_hourly_data(patient_id):
    """
    Hastaya saatlik veri ekler, tahmin yapar ve tahminin güvenilirliğini hesaplar.
    Hasta bilgileri (Yaş, Cinsiyet) ve saat bilgisi model girdisine enjekte edilir.
    """
    try:
        data = request.get_json() or {}

        if 'hour' not in data or 'vital_signs' not in data:
            return jsonify({'success': False, 'error': 'hour ve vital_signs gerekli'}), 400

        try:
            hour = int(data['hour'])
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Saat geçerli bir tam sayı olmalı'}), 400
        if hour < 0:
            return jsonify({'success': False, 'error': 'Saat negatif olamaz'}), 400

        # Validasyon
        is_valid, errors = validate_vital_signs(data['vital_signs'])
        if not is_valid:
            return jsonify({'success': False, 'error': 'Validation failed', 'details': errors}), 400

        vital_signs, transform_notes = normalize_inputs(data['vital_signs'])
        if not any(k not in MODEL_UNUSED_FIELDS for k in vital_signs):
            return jsonify({
                'success': False,
                'error': 'Hiç ölçüm girilmedi. Boş form modele gönderilirse eksikler "normal" '
                         'medyan değerlerle doldurulur ve sonuç yanıltıcı olur.'
            }), 400

        conn = get_db()

        patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
        if not patient:
            conn.close()
            return jsonify({'success': False, 'error': 'Hasta bulunamadı'}), 404

        # Statik veriler (Model Gender için Male=1, Female=0 bekler)
        demographics = {}
        if patient['gender']:
            demographics['Gender'] = 1 if patient['gender'] in ['Male', 'Erkek', 'male', 'erkek', '1'] else 0
        if patient['age']:
            demographics['Age'] = patient['age']

        prev_rows = conn.execute('''
            SELECT hour, vital_signs FROM hourly_data
            WHERE patient_id = ? AND hour < ?
            ORDER BY hour ASC
        ''', (patient_id, hour)).fetchall()

        history = []
        for row in prev_rows:
            row_dict = normalize_inputs(json.loads(row['vital_signs']))[0]
            row_dict.update(demographics, Hour=row['hour'])
            history.append(row_dict)

        current_dict = dict(vital_signs)
        current_dict.update(demographics, Hour=hour)
        history.append(current_dict)

        # Tahmin + güvenilirlik
        prediction, mc_samples, real_hours, real_rows = predict_with_history(history)
        assessment = estimator.assess(real_rows, real_hours, WINDOW_SIZE, mc_samples, transform_notes)
        if 'Age' not in demographics:
            assessment['reasons'].append('Yaş girilmemiş; eğitim verisinin medyan yaşı kullanıldı.')
        assessment['calibrated_probability'] = estimator.calibrate(prediction)
        assessment['unused_fields'] = [f for f in MODEL_UNUSED_FIELDS if f in vital_signs]

        if assessment['sufficient_data']:
            risk_text, risk_color, red_flags = get_risk_level(prediction, current_dict)
        else:
            risk_text, risk_color, red_flags = 'Yetersiz Veri', '#6b7280', []
        assessment['red_flags'] = red_flags
        apply_clinical_disagreement(assessment, prediction, OPTIMAL_THRESHOLD, red_flags)

        conn.execute('''
            INSERT OR REPLACE INTO hourly_data
            (patient_id, hour, vital_signs, prediction, risk_level, reliability, assessment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, hour, json.dumps(vital_signs), prediction, risk_text,
              assessment['score'], json.dumps(assessment, ensure_ascii=False)))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'hour': hour,
            'prediction': prediction,
            'risk_level': risk_text,
            'risk_color': risk_color,
            'is_sepsis_risk': assessment['sufficient_data'] and prediction >= OPTIMAL_THRESHOLD,
            'threshold': OPTIMAL_THRESHOLD,
            'reliability': assessment,
            'message': 'Veri başarıyla işlendi'
        }), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


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

        print(f"[OK] Hasta silindi: {patient['name']} (ID: {patient_id})")

        return jsonify({
            'success': True,
            'message': 'Hasta başarıyla silindi'
        }), 200

    except Exception as e:
        print(f"[FAIL] Hasta silme hatası: {str(e)}")
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
        'preprocessing_loaded': (transformer is not None) if FEATURE_SET == 'v3' else (imputer is not None and scaler is not None),
        'feature_set': FEATURE_SET,
        'calibrator_loaded': estimator is not None and estimator.calibrator is not None,
        'database_exists': os.path.exists(DB_PATH)
    }), 200


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Aktif modelin bilgileri ve ölçülmüş güvenilirlik istatistikleri"""
    stats = estimator.stats if estimator else {}
    return jsonify({
        'success': True,
        'model': MODEL_INFO,
        'threshold': OPTIMAL_THRESHOLD,
        'window_size': WINDOW_SIZE,
        'features': feature_names() if FEATURE_SET == 'v3' else numerical_columns,
        'feature_set': FEATURE_SET,
        'unused_fields': MODEL_UNUSED_FIELDS,
        'reliability_stats': stats,
    }), 200


# ============================================================================
# SERVER BAŞLATMA
# ============================================================================

def port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


if __name__ == '__main__':
    if port_in_use(PORT):
        print(f"\n[FAIL] {PORT} portu kullanimda. Uygulama zaten acik olabilir: http://localhost:{PORT}")
        print("       Eski pencereyi kapatip tekrar deneyin.")
        raise SystemExit(1)

    # Veritabanını başlat
    init_database()

    # Modeli yükle
    if load_model_and_preprocessing():
        # Model warm-up (ilk tahmin yavaş olmasın diye)
        print("\n[...] Model warm-up yapiliyor...")
        try:
            dummy_input = np.zeros((1, WINDOW_SIZE, model.input_shape[-1]), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            mc_dropout_predict(model, dummy_input)
            print("  [OK] Model warm-up tamamlandi")
        except Exception as e:
            print(f"  [WARN] Warm-up hatasi (onemli degil): {e}")

        url = f"http://localhost:{PORT}"
        print("\n" + "=" * 60)
        print(f"[WEB] Web arayuzu: {url}")
        print(f"[INFO] Model: {MODEL_INFO.get('name')} ({FEATURE_SET}), esik {OPTIMAL_THRESHOLD}")
        print("[INFO] Cikmak icin: CTRL+C")
        print("=" * 60 + "\n")

        if os.environ.get('SEPSIS_OPEN_BROWSER') == '1':
            import threading
            import webbrowser
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        # debug=False performansı artırır (TensorFlow reloading yapmaz)
        app.run(debug=False, host=HOST, port=PORT, threaded=True)
    else:
        print("\n[FAIL] Sunucu baslatilamadi. Model yukleme hatasi!")
        raise SystemExit(1)
