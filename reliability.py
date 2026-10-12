"""
Sepsis Tahmin Sistemi — Tahmin Güvenilirliği
=============================================

Model tek başına bir olasılık üretir; o olasılığın ne kadar güvenilir
olduğunu söylemez. Eksik alanlar eğitim setinin medyanlarıyla doldurulduğu
için hiç veri girilmediğinde model "ortalama bir YBÜ hastası" görür ve
düşük risk döndürür. Bu modül her tahmin için şunları hesaplar:

  1. Minimum veri kapısı  : Temel vital bulgular yoksa tahmin "Yetersiz Veri" olur.
  2. Veri tamlığı         : Ağırlıklı olarak gerçekten girilen (impute edilmeyen) değer oranı.
  3. Zamansal kapsam      : 6 saatlik pencerenin kaçı gerçek ölçüm (geri kalanı kopyalama).
  4. Model belirsizliği   : MC-Dropout ile N stokastik ileri geçiş → std, %90 aralık, karar uyumu.
  5. Dağılım dışılık      : Girilen değerlerin eğitim dağılımına uzaklığı (|z|).
  6. Kalibrasyon          : Validasyon setinde öğrenilmiş izotonik regresyon ile
                            ham skor → gözlenen sepsis oranı.

Sonuç 0-100 arası tek bir güvenilirlik skoru ve açıklayıcı nedenlerdir.
"""

import json
import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# Özellik grupları
# ---------------------------------------------------------------------------

# Tahmin için zorunlu kabul edilen temel vital bulgular (MAP, SBP/DBP'den türetilebilir)
CORE_VITALS = ['HR', 'O2Sat', 'Temp', 'Resp', 'MAP']
MIN_CORE_VITALS = 3

# Sepsis/SOFA açısından en bilgilendirici laboratuvar değerleri
# Formdaki hızlı girişle aynı 4 laboratuvar değeri
KEY_LABS = ['Lactate', 'WBC', 'Creatinine', 'Platelets']

OTHER_FEATURES = [
    'SBP', 'DBP', 'Bilirubin_total', 'pH', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Bilirubin_direct', 'Glucose', 'Magnesium',
    'Phosphate', 'Potassium', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'Fibrinogen',
]

# Tamlık skorundaki grup ağırlıkları (toplam 1)
GROUP_WEIGHTS = {'core': 0.70, 'labs': 0.20, 'other': 0.10}

# Birleşik güvenilirlik skorundaki bileşen ağırlıkları (toplam 1)
COMPONENT_WEIGHTS = {'completeness': 0.45, 'temporal': 0.20, 'certainty': 0.25, 'in_distribution': 0.10}

RELIABILITY_LEVELS = [
    (70, 'Yüksek', '#10b981'),
    (45, 'Orta', '#f59e0b'),
    (0, 'Düşük', '#ef4444'),
]

OOD_Z_LIMIT = 5.0
MC_SAMPLES = 30

FEATURE_LABELS = {
    'HR': 'Nabız', 'O2Sat': 'O2 Sat.', 'Temp': 'Ateş', 'Resp': 'Solunum', 'MAP': 'MAP',
    'SBP': 'Sistolik KB', 'DBP': 'Diyastolik KB', 'Lactate': 'Laktat', 'WBC': 'WBC',
    'Creatinine': 'Kreatinin', 'Platelets': 'Trombosit', 'Bilirubin_total': 'Total Bilirubin',
    'pH': 'pH', 'FiO2': 'FiO2',
}


def _label(feature):
    return FEATURE_LABELS.get(feature, feature)


def _has_value(row, key):
    value = row.get(key)
    if value is None or value == '':
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Girdi normalizasyonu
# ---------------------------------------------------------------------------

def normalize_inputs(vital_signs):
    """
    Kullanıcı girdisini eğitim verisinin birimlerine çevirir.
      - FiO2 eğitimde 0-1 oranıdır; 21-100 (%) girilirse 100'e bölünür.
      - MAP yoksa SBP/DBP'den hesaplanır: MAP = (SBP + 2·DBP) / 3
    Uygulanan dönüşümler not listesi olarak döner.
    """
    vs = {k: v for k, v in vital_signs.items() if v is not None and v != ''}
    notes = []

    if _has_value(vs, 'FiO2') and float(vs['FiO2']) > 1.0:
        vs['FiO2'] = round(float(vs['FiO2']) / 100.0, 3)
        notes.append(f"FiO2 % olarak girildi, orana çevrildi ({vs['FiO2']})")

    if not _has_value(vs, 'MAP') and _has_value(vs, 'SBP') and _has_value(vs, 'DBP'):
        vs['MAP'] = round((float(vs['SBP']) + 2 * float(vs['DBP'])) / 3.0, 1)
        notes.append(f"MAP, SBP/DBP'den hesaplandı ({vs['MAP']} mmHg)")

    return vs, notes


def core_vitals_present(row):
    return [f for f in CORE_VITALS if _has_value(row, f)]


# ---------------------------------------------------------------------------
# Kalibrasyon ve istatistikler
# ---------------------------------------------------------------------------

class ReliabilityEstimator:
    """Kalibratörü ve eğitim dağılımı istatistiklerini tutar."""

    def __init__(self, feature_stats, threshold, calibration_dir=None):
        """feature_stats: {sütun: (eğitim ortalaması, eğitim std)} — dağılım dışılık kontrolü için"""
        self.feature_stats = dict(feature_stats)
        self.threshold = float(threshold)
        self.calibrator = None
        self.stats = {}

        if calibration_dir:
            cal_path = os.path.join(calibration_dir, 'calibrator.pkl')
            stats_path = os.path.join(calibration_dir, 'reliability_stats.json')
            if os.path.exists(cal_path):
                with open(cal_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)

    # --- Kalibrasyon ------------------------------------------------------
    def calibrate(self, raw_prob):
        if self.calibrator is None:
            return None
        return float(self.calibrator.predict(np.array([raw_prob], dtype=float))[0])

    # --- 1. Veri tamlığı --------------------------------------------------
    def completeness(self, window_rows):
        """
        Pencere içindeki gerçek saatlerde her özelliğin en az bir kez
        ölçülüp ölçülmediğine bakar (lab değerleri her saat ölçülmez).
        Temel vitaller için son saat ağırlığı daha yüksektir.
        """
        if not window_rows:
            return 0.0, CORE_VITALS + KEY_LABS

        current = window_rows[-1]

        core_now = sum(_has_value(current, f) for f in CORE_VITALS) / len(CORE_VITALS)
        core_window = sum(any(_has_value(r, f) for r in window_rows) for f in CORE_VITALS) / len(CORE_VITALS)
        core_score = 0.7 * core_now + 0.3 * core_window

        labs_score = sum(any(_has_value(r, f) for r in window_rows) for f in KEY_LABS) / len(KEY_LABS)
        other_score = sum(any(_has_value(r, f) for r in window_rows) for f in OTHER_FEATURES) / len(OTHER_FEATURES)

        score = (GROUP_WEIGHTS['core'] * core_score
                 + GROUP_WEIGHTS['labs'] * labs_score
                 + GROUP_WEIGHTS['other'] * other_score)

        missing = [f for f in CORE_VITALS if not _has_value(current, f)]
        missing += [f for f in KEY_LABS if not any(_has_value(r, f) for r in window_rows)]
        return float(score), missing

    # --- 3. Dağılım dışılık ------------------------------------------------
    def out_of_distribution(self, current_row):
        flagged = []
        for col, (mean, std) in self.feature_stats.items():
            if col in ('Hour', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'):
                continue
            if not _has_value(current_row, col) or std == 0:
                continue
            z = (float(current_row[col]) - mean) / std
            if abs(z) > OOD_Z_LIMIT:
                flagged.append({'feature': col, 'value': float(current_row[col]), 'z': round(float(z), 1)})
        score = 1.0 - min(len(flagged) / 3.0, 1.0)
        return score, flagged

    # --- 4. Model belirsizliği --------------------------------------------
    def certainty(self, mc_samples):
        """
        MC-Dropout örneklerinin kaçının ortalama ile aynı tarafta (eşiğe göre)
        kaldığı → karar uyumu. 0.5 uyum = yazı-tura, 1.0 = tam uyum.
        """
        samples = np.asarray(mc_samples, dtype=float)
        mean = float(samples.mean())
        same_side = np.mean((samples >= self.threshold) == (mean >= self.threshold))
        score = float(np.clip((same_side - 0.5) / 0.5, 0.0, 1.0))
        return score, {
            'mean': mean,
            'std': float(samples.std()),
            'p05': float(np.percentile(samples, 5)),
            'p95': float(np.percentile(samples, 95)),
            'agreement': float(same_side),
            'n': int(len(samples)),
        }

    # --- Birleşik değerlendirme --------------------------------------------
    def assess(self, window_rows, real_hours, window_size, mc_samples, transform_notes=None):
        current = window_rows[-1] if window_rows else {}
        core_present = core_vitals_present(current)
        sufficient = len(core_present) >= MIN_CORE_VITALS

        completeness, missing = self.completeness(window_rows)
        temporal = min(real_hours, window_size) / float(window_size)
        certainty, mc = self.certainty(mc_samples)
        in_dist, ood = self.out_of_distribution(current)

        score = 100.0 * (COMPONENT_WEIGHTS['completeness'] * completeness
                         + COMPONENT_WEIGHTS['temporal'] * temporal
                         + COMPONENT_WEIGHTS['certainty'] * certainty
                         + COMPONENT_WEIGHTS['in_distribution'] * in_dist)
        if not sufficient:
            score = min(score, 25.0)

        level, color = next((lbl, col) for lim, lbl, col in RELIABILITY_LEVELS if score >= lim)

        reasons = []
        if not sufficient:
            reasons.append(
                f"Temel vital bulgulardan yalnızca {len(core_present)}/{len(CORE_VITALS)} girildi "
                f"(en az {MIN_CORE_VITALS} gerekli). Eksikler model tarafından 'normal' "
                f"medyan değerlerle doldurulur; sonuç hastayı yansıtmaz."
            )
        missing_core = [f for f in missing if f in CORE_VITALS]
        missing_labs = [f for f in missing if f in KEY_LABS]
        if sufficient and missing_core:
            reasons.append('Eksik vital: ' + ', '.join(_label(f) for f in missing_core))
        if missing_labs:
            reasons.append('Son 6 saatte ölçülmemiş kritik lab: ' + ', '.join(_label(f) for f in missing_labs))
        if real_hours < window_size:
            reasons.append(
                f"Model 6 saatlik pencere bekliyor; {real_hours} saat gerçek veri var, "
                f"kalan {window_size - real_hours} saat ilk ölçümün kopyası."
            )
        if mc['agreement'] < 0.9:
            reasons.append(
                f"Model kararsız: stokastik geçişlerin %{mc['agreement'] * 100:.0f}'i aynı kararı verdi "
                f"(skor aralığı %{mc['p05'] * 100:.0f}–%{mc['p95'] * 100:.0f})."
            )
        for item in ood:
            reasons.append(
                f"{_label(item['feature'])} = {item['value']:g} eğitim verisinde çok nadir "
                f"(z={item['z']}); model bu bölgede az örnek gördü."
            )
        for note in transform_notes or []:
            reasons.append(note)

        return {
            'score': round(score, 1),
            'level': level if sufficient else 'Yetersiz Veri',
            'color': color if sufficient else '#6b7280',
            'sufficient_data': sufficient,
            'components': {
                'completeness': round(completeness * 100, 1),
                'temporal': round(temporal * 100, 1),
                'certainty': round(certainty * 100, 1),
                'in_distribution': round(in_dist * 100, 1),
            },
            'core_vitals_present': core_present,
            'missing': missing,
            'out_of_distribution': ood,
            'mc_dropout': mc,
            'reasons': reasons,
        }


def apply_clinical_disagreement(assessment, prediction, threshold, red_flags, min_points=2, cap=55.0):
    """
    Klinik kırmızı bayraklar belirgin (>= min_points) iken model skoru eşiğin
    altındaysa model ile klinik tablo çelişir. Bu durumda risk sınıfını
    klinik kurallar belirler; model skorunun güvenilirliği en fazla 'Orta' olur.
    """
    points = sum(f['points'] for f in red_flags)
    if points < min_points or prediction >= threshold:
        assessment['clinical_disagreement'] = False
        return assessment

    assessment['clinical_disagreement'] = True
    assessment['reasons'].insert(0,
        f"Model skoru (%{prediction * 100:.1f}) klinik tabloyla çelişiyor: {points} puanlık kırmızı bayrak var. "
        f"Risk sınıfını klinik kurallar belirledi; bu alarmı ciddiye alın."
    )
    if assessment['score'] > cap:
        assessment['score'] = cap
        level, color = next((lbl, col) for lim, lbl, col in RELIABILITY_LEVELS if cap >= lim)
        assessment['level'], assessment['color'] = level, color
    return assessment


# ---------------------------------------------------------------------------
# MC-Dropout
# ---------------------------------------------------------------------------

def mc_dropout_predict(model, x_seq, n_samples=MC_SAMPLES):
    """
    Dropout katmanları açık, BatchNorm çıkarım modunda olacak şekilde
    modeli N kez çalıştırır. Model Sequential/zincir yapıda varsayılır
    (models/optimized/best_model.keras böyledir); değilse deterministik
    tahmini N kez döndürür ve belirsizlik 0 görünür.
    """
    import tensorflow as tf

    batch = np.repeat(x_seq.astype(np.float32), n_samples, axis=0)
    try:
        h = tf.convert_to_tensor(batch)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if isinstance(layer, tf.keras.layers.GaussianNoise):
                continue  # gürültü yalnızca eğitim içindir, belirsizlik ölçümüne katılmaz
            if isinstance(layer, tf.keras.layers.Dropout):
                h = layer(h, training=True)
            else:
                h = layer(h, training=False)
        return np.asarray(h).reshape(-1)
    except Exception:
        p = float(model.predict(x_seq, verbose=0).reshape(-1)[0])
        return np.full(n_samples, p)
