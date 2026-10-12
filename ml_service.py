"""
Sepsis Tahmin Sistemi — Model Servisi
=====================================

Modeli ve ön işleme nesnelerini yükler, bir hastanın saatlik geçmişinden
risk tahmini + güvenilirlik değerlendirmesi üretir. Web katmanından
(app.py) ve veritabanından bağımsızdır.
"""

import os
import pickle
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from config import (MODEL_PATH, PREPROCESSING_DIR, OPTIMAL_THRESHOLD, RISK_LEVELS,
                    CLINICAL_OVERRIDES, FEATURE_SET)
from features import build_hourly_features, make_windows
from reliability import (ReliabilityEstimator, apply_clinical_disagreement, mc_dropout_predict,
                         normalize_inputs)

WINDOW_SIZE = 6

# Modelin girdisinde kullanılmayan (eğitimde hariç tutulan) alanlar — yalnızca kayıt amaçlı
MODEL_UNUSED_FIELDS = ['PCT', 'CRP', 'Presepsin', 'IL6']

model = None
imputer = None
scaler = None
ohe = None
transformer = None
feature_stats = {}
estimator = None
numerical_columns = []
categorical_columns = []


# ============================================================================
# ESKİ MODELLER İÇİN ÖZEL KATMANLAR (legacy modelin yüklenebilmesi için)
# ============================================================================

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=0.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        return K.mean(weight * cross_entropy, axis=1)
    return binary_focal_loss_fixed


class AttentionLayer(layers.Layer):
    """Temporal Attention Mechanism (eski modeller)"""

    def build(self, input_shape):
        self.time_steps = input_shape[1]
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='normal', trainable=True)
        if self.time_steps is not None:
            self.b = self.add_weight(name='attention_bias', shape=(self.time_steps, 1),
                                     initializer='zeros', trainable=True)
        else:
            self.b = None
        super().build(input_shape)

    def call(self, x):
        e = K.dot(x, self.W)
        if self.b is not None:
            e += self.b
        a = K.softmax(K.tanh(e), axis=1)
        return K.sum(x * a, axis=1)


# ============================================================================
# GİRDİ DOĞRULAMA
# ============================================================================

VITAL_SIGN_RANGES = {
    'HR': (40, 200), 'Temp': (35, 42), 'SBP': (60, 250), 'DBP': (30, 150), 'MAP': (40, 180),
    'Resp': (8, 50), 'O2Sat': (70, 100), 'EtCO2': (10, 80),
    'WBC': (1, 50), 'Platelets': (20, 800), 'Hgb': (5, 20), 'Hct': (15, 65),
    'Creatinine': (0.3, 15), 'BUN': (3, 150), 'Glucose': (30, 600), 'Lactate': (0.5, 20),
    'Bilirubin_total': (0.1, 30), 'Bilirubin_direct': (0, 15),
    'pH': (6.8, 7.8), 'PaCO2': (15, 100), 'PaO2': (40, 500), 'HCO3': (10, 45), 'BaseExcess': (-20, 20),
    'Calcium': (5, 15), 'Chloride': (70, 130), 'Potassium': (2, 8), 'Magnesium': (0.5, 5),
    'AST': (5, 5000), 'ALT': (5, 5000), 'ALP': (20, 1000),
    'PCT': (0.01, 100), 'CRP': (0, 500), 'Presepsin': (100, 5000), 'IL6': (0, 1000),
    'FiO2': (0.21, 100),          # 0.21-1.0 oran veya 21-100 (%) kabul edilir
    'SaO2': (50, 100), 'Alkalinephos': (10, 2000), 'Phosphate': (0.5, 15), 'PTT': (15, 150),
    'Fibrinogen': (30, 1200), 'TroponinI': (0, 100),
}


def validate_vital_signs(vital_signs):
    """Değerlerin sayı ve kabul edilen aralıkta olduğunu kontrol eder. (geçerli mi, hatalar)"""
    errors = []
    for field, value in vital_signs.items():
        if value is None or value == '':
            continue
        if field not in VITAL_SIGN_RANGES:
            errors.append(f'{field}: bilinmeyen alan')
            continue
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            errors.append(f'{field}: sayı olmalı')
            continue
        lo, hi = VITAL_SIGN_RANGES[field]
        if not lo <= num_value <= hi:
            errors.append(f'{field}: {lo}–{hi} aralığında olmalı (girilen {num_value:g})')
    return len(errors) == 0, errors


# ============================================================================
# MODEL YÜKLEME
# ============================================================================

def load_model_and_preprocessing():
    """Modeli ve ön işleme nesnelerini yükle"""
    global model, estimator
    try:
        print("\n" + "=" * 60)
        print("MODEL VE PREPROCESSING YUKLENIYOR...")
        print("=" * 60)
        print(f"\n[1/2] Model yukleniyor ({FEATURE_SET}): {MODEL_PATH}")

        custom_objects = {
            'binary_focal_loss': binary_focal_loss,
            'binary_focal_loss_fixed': binary_focal_loss(gamma=2.0, alpha=0.25),
            'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25),
            'AttentionLayer': AttentionLayer,
        }
        try:
            model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        except Exception:
            model = keras.models.load_model(MODEL_PATH)
        print("  [OK] Model basariyla yuklendi")

        if FEATURE_SET == 'v3':
            _load_v3_preprocessing()
        else:
            _load_legacy_preprocessing()

        estimator = ReliabilityEstimator(feature_stats, OPTIMAL_THRESHOLD,
                                         calibration_dir=os.path.dirname(MODEL_PATH))
        if estimator.calibrator is None:
            print("  [WARN] Kalibrator yok - 'python scripts/fit_calibration.py' calistirin")
        else:
            print("  [OK] Kalibrator ve guvenilirlik istatistikleri yuklendi")

        print("\n" + "=" * 60)
        print("[OK] TUM BILESENLER BASARIYLA YUKLENDI!")
        print("=" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] HATA: Model yuklenemedi! {e}")
        traceback.print_exc()
        return False


def _load_v3_preprocessing():
    """v3: ileri doldurma + ölçüm maskeleri (features.py), eğitim hastalarıyla fit edilmiş dönüştürücü"""
    global transformer, feature_stats
    print("\n[2/2] v3 ozellik donusturucu yukleniyor...")
    with open(os.path.join(os.path.dirname(MODEL_PATH), 'transformer.pkl'), 'rb') as f:
        transformer = pickle.load(f)
    feature_stats = transformer.clinical_stats()
    print(f"  [OK] {len(transformer.names)} ozellik (34 deger + 34 olcum maskesi + statik)")


def _load_legacy_preprocessing():
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


def warm_up():
    dummy = np.zeros((1, WINDOW_SIZE, model.input_shape[-1]), dtype=np.float32)
    model.predict(dummy, verbose=0)
    mc_dropout_predict(model, dummy)


def is_ready():
    preprocessing = (transformer is not None) if FEATURE_SET == 'v3' else (imputer is not None and scaler is not None)
    return model is not None and preprocessing


# ============================================================================
# TAHMİN
# ============================================================================

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
        X_seq = _build_legacy_sequence(real_rows, window_size)

    expected = model.input_shape[-1]
    if X_seq.shape[-1] != expected:
        raise ValueError(f'Özellik sayısı uyumsuz: {X_seq.shape[-1]} != model {expected}')
    return X_seq.astype(np.float32), real_hours, real_rows


def _build_legacy_sequence(real_rows, window_size):
    padding = [real_rows[0]] * (window_size - len(real_rows))
    df = pd.DataFrame(padding + real_rows)
    for col in numerical_columns:
        if col not in df.columns:
            df[col] = np.nan
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = ''
    df = df[numerical_columns + categorical_columns]
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    X_numerical = scaler.transform(imputer.transform(df[numerical_columns]))
    if categorical_columns and ohe is not None:
        X_combined = np.hstack([X_numerical, ohe.transform(df[categorical_columns])])
    else:
        X_combined = X_numerical
    return X_combined.reshape(1, window_size, -1)


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
    adjusted = prediction
    if points >= 5:
        adjusted = max(prediction, 0.60)
    elif points >= 3:
        adjusted = max(prediction, 0.35)
    elif points >= 2:
        adjusted = max(prediction, 0.25)

    for key in ('critical', 'high', 'moderate', 'low', 'very_low'):
        level = RISK_LEVELS[key]
        if adjusted >= level['min']:
            return level['label'], level['color'], flags
    return RISK_LEVELS['very_low']['label'], RISK_LEVELS['very_low']['color'], flags


def assess(previous_hours, hour, vital_signs, demographics, transform_notes=None):
    """
    Bir saatlik yeni ölçüm için tahmin + güvenilirlik.

    previous_hours: [(saat, {değişken: değer}), ...] — bu saatten önceki kayıtlar
    vital_signs:    normalize edilmiş bu saatin değerleri
    demographics:   {'Age': .., 'Gender': 0/1} (bilinmeyenler eksik bırakılır)
    """
    history = []
    for prev_hour, values in previous_hours:
        row = normalize_inputs(values)[0]
        row.update(demographics, Hour=prev_hour)
        history.append(row)
    current = dict(vital_signs)
    current.update(demographics, Hour=hour)
    history.append(current)

    X_seq, real_hours, real_rows = build_sequence(history)
    prediction = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
    mc_samples = mc_dropout_predict(model, X_seq)

    assessment = estimator.assess(real_rows, real_hours, WINDOW_SIZE, mc_samples, transform_notes)
    if 'Age' not in demographics:
        assessment['reasons'].append('Yaş girilmemiş; eğitim verisinin medyan yaşı kullanıldı.')
    assessment['calibrated_probability'] = estimator.calibrate(prediction)
    assessment['unused_fields'] = [f for f in MODEL_UNUSED_FIELDS if f in vital_signs]

    if assessment['sufficient_data']:
        risk_text, risk_color, red_flags = get_risk_level(prediction, current)
    else:
        risk_text, risk_color, red_flags = 'Yetersiz Veri', '#6b7280', []
    assessment['red_flags'] = red_flags
    apply_clinical_disagreement(assessment, prediction, OPTIMAL_THRESHOLD, red_flags)

    return {
        'prediction': prediction,
        'risk_level': risk_text,
        'risk_color': risk_color,
        'is_sepsis_risk': assessment['sufficient_data'] and prediction >= OPTIMAL_THRESHOLD,
        'assessment': assessment,
    }
