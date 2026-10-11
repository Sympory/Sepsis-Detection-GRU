"""
Eski (legacy) model için kalibrasyon ve güvenilirlik istatistiklerini üretir.

  - Validasyon setinde izotonik regresyon (ham skor → gözlenen sepsis oranı)
  - Test setinde ECE / Brier (ham vs kalibre), eşik metrikleri, karışıklık matrisi
  - Güvenilirlik diyagramı (reliability diagram)
  - Boş girdi (tüm alanlar medyan) için modelin verdiği skor

Çıktılar (config.MODEL_DIR altında):
  calibrator.pkl, reliability_stats.json, reliability_diagram.png

Kullanım:
  python scripts/fit_calibration.py
"""

import json
import os
import pickle
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

# Bu betik eski (legacy) model içindir; v3 kendi kalibrasyonunu train_v3.py'de yapar
from config import LEGACY_MODEL_PATH as MODEL_PATH, LEGACY_THRESHOLD as OPTIMAL_THRESHOLD, PREPROCESSING_DIR
from reliability import mc_dropout_predict


def load_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location('sepsis_app', os.path.join(ROOT, 'app.py'))
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    from tensorflow import keras
    custom_objects = {
        'AttentionLayer': app_module.AttentionLayer,
        'binary_focal_loss_fixed': app_module.binary_focal_loss(),
        'focal_loss_fixed': app_module.focal_loss(),
    }
    return keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)


def predict(model, path):
    X = np.load(path, mmap_mode='r')
    out = []
    for i in range(0, len(X), 16384):
        out.append(model.predict(np.asarray(X[i:i + 16384], dtype=np.float32), verbose=0).reshape(-1))
    return np.concatenate(out)


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_curve(y, p, bins=10):
    # Eşit sayıda örnek içeren kantil bölmeler (dengesiz veride daha anlamlı)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return [{'mean_pred': float(p[c].mean()), 'observed': float(y[c].mean()), 'n': int(len(c))} for c in chunks]


def threshold_metrics(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0.0
    return {'threshold': float(t), 'sensitivity': sens, 'specificity': spec, 'ppv': ppv, 'npv': npv, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    out_dir = os.path.dirname(MODEL_PATH)
    print(f'Model: {MODEL_PATH}')
    model = load_model()

    print('Validasyon tahminleri...')
    y_val = np.load(os.path.join(PREPROCESSING_DIR, 'y_val.npy'))
    p_val = predict(model, os.path.join(PREPROCESSING_DIR, 'X_val.npy'))

    print('Test tahminleri...')
    y_test = np.load(os.path.join(PREPROCESSING_DIR, 'y_test.npy'))
    p_test = predict(model, os.path.join(PREPROCESSING_DIR, 'X_test.npy'))

    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(p_val, y_val)
    p_test_cal = calibrator.predict(p_test)

    # Validasyonda F1'i maksimize eden eşik (test setine bakmadan seçilir)
    grid = np.linspace(0.02, 0.8, 79)
    f1s = [threshold_metrics(y_val, p_val, t)['f1'] for t in grid]
    best_val_t = float(grid[int(np.argmax(f1s))])
    # Tarama amaçlı: validasyonda en az %80 duyarlılık veren en yüksek eşik
    sens80_t = float(np.quantile(p_val[y_val == 1], 0.20))

    # Boş girdi: tüm özellikler medyan → ölçeklenmiş vektör
    with open(os.path.join(PREPROCESSING_DIR, 'imputer.pkl'), 'rb') as f:
        imputer = pickle.load(f)
    with open(os.path.join(PREPROCESSING_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    empty = scaler.transform(imputer.statistics_.reshape(1, -1)).astype(np.float32)
    empty_seq = np.repeat(empty[None, :, :], 6, axis=1)
    empty_raw = float(model.predict(empty_seq, verbose=0).reshape(-1)[0])

    # MC-Dropout: rastgele test örneklerinde belirsizlik ile hata ilişkisi
    rng = np.random.default_rng(42)
    X_test = np.load(os.path.join(PREPROCESSING_DIR, 'X_test.npy'), mmap_mode='r')
    pos = np.where(y_test == 1)[0]
    neg = np.where(y_test == 0)[0]
    sample_idx = np.concatenate([rng.choice(pos, 150, replace=False), rng.choice(neg, 150, replace=False)])
    agreements, correct = [], []
    for i in sample_idx:
        s = mc_dropout_predict(model, np.asarray(X_test[i:i + 1], dtype=np.float32))
        decision = s.mean() >= OPTIMAL_THRESHOLD
        agreements.append(np.mean((s >= OPTIMAL_THRESHOLD) == decision))
        correct.append(decision == bool(y_test[i]))
    agreements, correct = np.array(agreements), np.array(correct)
    hi = agreements >= 0.9

    stats = {
        'model_path': os.path.relpath(MODEL_PATH, ROOT).replace('\\', '/'),
        'prevalence_test': float(y_test.mean()),
        'n_val': int(len(y_val)),
        'n_test': int(len(y_test)),
        'test': {
            'roc_auc': float(roc_auc_score(y_test, p_test)),
            'pr_auc': float(average_precision_score(y_test, p_test)),
            'brier_raw': float(brier_score_loss(y_test, p_test)),
            'brier_calibrated': float(brier_score_loss(y_test, p_test_cal)),
            'ece_raw': ece(y_test, p_test),
            'ece_calibrated': ece(y_test, p_test_cal),
            'at_config_threshold': threshold_metrics(y_test, p_test, OPTIMAL_THRESHOLD),
            'at_val_best_f1_threshold': threshold_metrics(y_test, p_test, best_val_t),
            'at_val_sens80_threshold': threshold_metrics(y_test, p_test, sens80_t),
        },
        'val_best_f1_threshold': best_val_t,
        'val_sens80_threshold': sens80_t,
        'reliability_curve_raw': reliability_curve(y_test, p_test),
        'reliability_curve_calibrated': reliability_curve(y_test, p_test_cal),
        'empty_input': {'raw': empty_raw, 'calibrated': float(calibrator.predict([empty_raw])[0])},
        'mc_dropout_check': {
            'n': int(len(sample_idx)),
            'accuracy_high_agreement': float(correct[hi].mean()) if hi.any() else None,
            'accuracy_low_agreement': float(correct[~hi].mean()) if (~hi).any() else None,
            'share_low_agreement': float((~hi).mean()),
        },
    }

    with open(os.path.join(out_dir, 'calibrator.pkl'), 'wb') as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(out_dir, 'reliability_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for key, label in (('reliability_curve_raw', 'Ham skor'), ('reliability_curve_calibrated', 'İzotonik kalibre')):
            pts = stats[key]
            ax.plot([q['mean_pred'] for q in pts], [q['observed'] for q in pts], marker='o', label=label)
        lim = max(max(q['mean_pred'] for q in stats['reliability_curve_raw']), 0.1)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='Mükemmel kalibrasyon')
        ax.set_xlabel('Ortalama tahmin')
        ax.set_ylabel('Gözlenen sepsis oranı')
        ax.set_title('Güvenilirlik Diyagramı (test)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'reliability_diagram.png'), dpi=130)
    except Exception as e:
        print(f'Grafik çizilemedi: {e}')

    print(json.dumps({k: stats[k] for k in ('test', 'val_best_f1_threshold', 'val_sens80_threshold', 'empty_input', 'mc_dropout_check')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
