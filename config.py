"""
Sepsis Tahmin Sistemi — Merkezi Konfigürasyon
==============================================

Tüm dosyalar bu config'i kullanarak model yolu, threshold, 
feature listesi gibi değerlere erişir.
"""

import os

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------- MODEL ----------
# Aktif model: models/v3 (ileri doldurma + ölçüm maskeleri) varsayılandır.
# Eski model için:  set SEPSIS_MODEL=legacy   (Windows)  /  export SEPSIS_MODEL=legacy
import json

V3_DIR = os.path.join(PROJECT_ROOT, 'models', 'v3')

# --- Eski model (Optimized v1, 40 özellik, medyan doldurma) ---
LEGACY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'optimized', 'best_model.keras')
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Optimal threshold (F1 bazlı: compare_models_full.py sonucu)
LEGACY_THRESHOLD = 0.24

# Model bilgileri (UI'da göstermek için)
LEGACY_MODEL_INFO = {
    # Keras Tuner en iyi denemede attention KULLANMADI (use_attention=False):
    # mimari 2 katmanlı Bi-GRU + Dense'tir.
    'name': 'Bi-GRU (2 katman) + Dense',
    'version': 'Optimized v1',
    'params': 124545,
    'roc_auc': 0.8086,
    'pr_auc': 0.0892,
    'best_f1': 0.1677,
    'threshold': 0.24,
    'input_shape': (6, 40),
    'loss_function': 'Focal Loss (γ=2, α=0.25)',
    'trained_date': '2026',
    'architecture': [
        'Input(6 saat × 40 özellik)',
        'Bidirectional GRU(64) → 128',
        'BatchNorm → Dropout(0.4)',
        'Bidirectional GRU(64) → 128',
        'BatchNorm → Dropout(0.4)',
        'Dense(64, ReLU) → Dropout(0.4)',
        'Dense(1, Sigmoid) → P(Sepsis)',
    ],
}


def _load_v3_card():
    card_path = os.path.join(V3_DIR, 'model_card.json')
    if not (os.path.exists(card_path) and os.path.exists(os.path.join(V3_DIR, 'model.keras'))
            and os.path.exists(os.path.join(V3_DIR, 'transformer.pkl'))):
        return None
    with open(card_path, encoding='utf-8') as f:
        return json.load(f)


_requested = os.environ.get('SEPSIS_MODEL', 'v3').lower()
_v3_card = _load_v3_card() if _requested == 'v3' else None

if _v3_card:
    FEATURE_SET = 'v3'
    MODEL_PATH = os.path.join(V3_DIR, 'model.keras')
    OPTIMAL_THRESHOLD = round(float(_v3_card['threshold']), 3)
    MODEL_INFO = dict(_v3_card, threshold=OPTIMAL_THRESHOLD)
else:
    FEATURE_SET = 'legacy'
    MODEL_PATH = LEGACY_MODEL_PATH
    OPTIMAL_THRESHOLD = LEGACY_THRESHOLD
    MODEL_INFO = LEGACY_MODEL_INFO

# ---------- VERİ ----------
SEQUENCE_WINDOW = 6
NUM_FEATURES = 40

NUMERICAL_COLUMNS = [
    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'
]

# ---------- RISK SEVİYELERİ ----------
RISK_LEVELS = {
    'very_low':  {'min': 0.00, 'max': 0.10, 'label': 'Çok Düşük', 'color': '#10b981', 'emoji': '🟢'},
    'low':       {'min': 0.10, 'max': 0.20, 'label': 'Düşük',      'color': '#3b82f6', 'emoji': '🔵'},
    'moderate':  {'min': 0.20, 'max': 0.35, 'label': 'Orta',        'color': '#f59e0b', 'emoji': '🟠'},
    'high':      {'min': 0.35, 'max': 0.55, 'label': 'Yüksek',     'color': '#ef4444', 'emoji': '🔴'},
    'critical':  {'min': 0.55, 'max': 1.00, 'label': 'Çok Yüksek', 'color': '#7f1d1d', 'emoji': '🚨'},
}

# ---------- KLİNİK OVERRIDE ----------
CLINICAL_OVERRIDES = {
    'SBP_low': {'threshold': 90, 'direction': 'below', 'red_flags': 1},
    'MAP_low': {'threshold': 65, 'direction': 'below', 'red_flags': 1},
    'O2Sat_low': {'threshold': 90, 'direction': 'below', 'red_flags': 1},
    'Temp_high': {'threshold': 38.5, 'direction': 'above', 'red_flags': 1},
    'Temp_low': {'threshold': 35.5, 'direction': 'below', 'red_flags': 1},
    'HR_high': {'threshold': 120, 'direction': 'above', 'red_flags': 1},
    'Lactate_high': {'threshold': 2.0, 'direction': 'above', 'red_flags': 2},
    'pH_low': {'threshold': 7.30, 'direction': 'below', 'red_flags': 1},
    'WBC_high': {'threshold': 12, 'direction': 'above', 'red_flags': 1},
    'WBC_low': {'threshold': 4, 'direction': 'below', 'red_flags': 1},
}

# ---------- VERİTABANI ----------
DB_PATH = os.environ.get('SEPSIS_DB_PATH', os.path.join(PROJECT_ROOT, 'patients.db'))

# ---------- SUNUCU ----------
HOST = os.environ.get('SEPSIS_HOST', '127.0.0.1')
PORT = 5000
DEBUG = False
