"""
Sepsis GRU v3 — eğitim, kalibrasyon ve değerlendirme
====================================================

Önceki sürümlere göre farklar:
  - İleri doldurma + ölçüm maskeleri (features.py)
  - Hasta bazlı, sepsis durumuna göre tabakalı 60/20/20 bölme
  - Ölçekleyici/medyanlar YALNIZCA eğitim hastalarıyla fit edilir
  - İlk 5 saat de eğitime dahil (uygulama 1. saatten tahmin yapıyor)
  - Düz BCE + sınıf ağırlığı, ardından izotonik kalibrasyon
  - Klinik tutarlılık testi (LightGBM karşılaştırması: scripts/baseline_lightgbm.py)

Çıktılar: models/v3/
  model.keras, transformer.pkl, calibrator.pkl, reliability_stats.json,
  model_card.json, training_history.json, figures (*.png)

Kullanım:
  python scripts/train_v3.py --epochs 20
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)

from features import (CLINICAL_COLUMNS, FeatureTransformer, build_hourly_features,
                      feature_names, make_windows)
from fit_calibration import ece, reliability_curve, threshold_metrics

WINDOW = 6


def split_patients(df, seed=42):
    """Sepsis olan/olmayan hastalar ayrı ayrı karıştırılıp 60/20/20 bölünür."""
    status = df.groupby('Patient_ID')['SepsisLabel'].max()
    rng = np.random.default_rng(seed)
    parts = {'train': [], 'val': [], 'test': []}
    for label in (0, 1):
        ids = status[status == label].index.to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        parts['test'] += list(ids[:int(0.2 * n)])
        parts['val'] += list(ids[int(0.2 * n):int(0.4 * n)])
        parts['train'] += list(ids[int(0.4 * n):])
    return {k: set(v) for k, v in parts.items()}


def windows_for(df, X, ids):
    xs, ys, hours, groups = [], [], [], []
    for pid, idx in df.groupby('Patient_ID', sort=False).indices.items():
        if pid not in ids:
            continue
        idx = np.sort(idx)
        xs.append(make_windows(X[idx], WINDOW))
        ys.append(df['SepsisLabel'].to_numpy()[idx])
        hours.append(np.arange(len(idx)))
        groups.append(np.full(len(idx), pid))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(hours), np.concatenate(groups))


def build_model(n_features):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    l2 = regularizers.l2(1e-4)
    inputs = keras.Input(shape=(WINDOW, n_features))
    x = layers.GaussianNoise(0.1, name='noise')(inputs)  # yalnızca eğitimde aktif
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, kernel_regularizer=l2), name='bigru')(x)
    x = layers.Dropout(0.4, name='drop_1')(x)
    x = layers.GRU(64, kernel_regularizer=l2, name='gru')(x)
    x = layers.Dropout(0.4, name='drop_2')(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2, name='dense')(x)
    x = layers.Dropout(0.4, name='drop_3')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs, outputs, name='sepsis_gru_v3')
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='roc_auc'), keras.metrics.AUC(curve='PR', name='pr_auc')],
    )
    return model


def clinical_sanity(model, transformer):
    """
    Aynı saatte 'normal' ve 'septik' vital bulgulu iki hasta: septik olan
    daha yüksek skor almalı. Ayrıca tek tek kötüleşen parametreler skoru
    düşürmemeli (monotonluk).
    """
    normal = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75, 'MAP': 92,
              'Lactate': 1.2, 'WBC': 8, 'Age': 65, 'Gender': 1}
    septic = dict(normal, HR=130, O2Sat=88, Temp=38.9, Resp=28, SBP=82, DBP=48, MAP=59, Lactate=4.5, WBC=17)

    def score(values, hour):
        rows = [dict(values, Hour=hour - WINDOW + 1 + i) for i in range(WINDOW)]
        X = transformer.transform(build_hourly_features(pd.DataFrame(rows)))
        return float(model.predict(X[None], verbose=0)[0, 0])

    out = {'normal_vs_septic': {}, 'single_feature': {}}
    for hour in (6, 24, 72):
        out['normal_vs_septic'][hour] = {'normal': score(normal, hour), 'septic': score(septic, hour)}
    base = score(normal, 24)
    for feat, bad in (('HR', 135), ('Temp', 39.2), ('Resp', 30), ('MAP', 58), ('SBP', 80),
                      ('O2Sat', 87), ('Lactate', 5.0), ('WBC', 20)):
        out['single_feature'][feat] = {'value': bad, 'delta': score(dict(normal, **{feat: bad}), 24) - base}
    out['passed'] = (all(v['septic'] > v['normal'] for v in out['normal_vs_septic'].values())
                     and sum(v['delta'] >= 0 for v in out['single_feature'].values()) >= 6)
    return out


def save_figures(out_dir, history, y, p, p_cal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ep = range(1, len(history['loss']) + 1)
    axes[0].plot(ep, history['loss'], label='Eğitim'); axes[0].plot(ep, history['val_loss'], label='Validasyon')
    axes[0].set_title('Kayıp'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(ep, history['roc_auc'], label='Eğitim'); axes[1].plot(ep, history['val_roc_auc'], label='Validasyon')
    axes[1].set_title('ROC-AUC'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    axes[2].plot(ep, history['pr_auc'], label='Eğitim'); axes[2].plot(ep, history['val_pr_auc'], label='Validasyon')
    axes[2].set_title('PR-AUC'); axes[2].set_xlabel('Epoch'); axes[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'training_history.png'), dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, p):.3f}'); axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('ROC eğrisi (test)'); axes[0].set_xlabel('Yanlış pozitif oranı'); axes[0].set_ylabel('Duyarlılık'); axes[0].legend()
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, label=f'PR-AUC = {average_precision_score(y, p):.3f}')
    axes[1].axhline(y.mean(), color='k', ls='--', lw=1, label=f'Rastgele = {y.mean():.3f}')
    axes[1].set_title('Precision-Recall (test)'); axes[1].set_xlabel('Duyarlılık'); axes[1].set_ylabel('Kesinlik'); axes[1].legend()
    for curve, label in ((reliability_curve(y, p), 'Ham'), (reliability_curve(y, p_cal), 'Kalibre')):
        axes[2].plot([c['mean_pred'] for c in curve], [c['observed'] for c in curve], marker='o', label=label)
    lim = max(c['mean_pred'] for c in reliability_curve(y, p))
    axes[2].plot([0, lim], [0, lim], 'k--', lw=1, label='İdeal')
    axes[2].set_title('Güvenilirlik diyagramı'); axes[2].set_xlabel('Ortalama tahmin'); axes[2].set_ylabel('Gözlenen oran'); axes[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'evaluation.png'), dpi=120); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(ROOT, 'data', 'train.csv'))
    parser.add_argument('--output', default=os.path.join(ROOT, 'models', 'v3'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--pos-weight', type=float, default=5.0)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)

    print('[1/7] Veri okunuyor...')
    df = pd.read_csv(args.input)
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True)
    df = df.dropna(subset=['SepsisLabel'])
    print(f'  {len(df):,} saat, {df.Patient_ID.nunique():,} hasta')

    print('[2/7] Hasta bazlı bölme ve özellikler...')
    parts = split_patients(df)
    feats = build_hourly_features(df, group_col='Patient_ID')
    transformer = FeatureTransformer().fit(feats[df['Patient_ID'].isin(parts['train']).to_numpy()])
    X_all = transformer.transform(feats)
    del feats

    print('[3/7] Pencereler...')
    Xtr, ytr, _, _ = windows_for(df, X_all, parts['train'])
    Xva, yva, _, _ = windows_for(df, X_all, parts['val'])
    Xte, yte, hte, gte = windows_for(df, X_all, parts['test'])
    del X_all
    for name, y in (('train', ytr), ('val', yva), ('test', yte)):
        print(f'  {name}: {len(y):,} pencere, %{y.mean() * 100:.2f} pozitif')

    print('[4/7] Eğitim...')
    model = build_model(Xtr.shape[-1])
    model.summary(print_fn=lambda s: print('  ' + s))
    ckpt = os.path.join(args.output, 'model.keras')
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch_size,
        class_weight={0: 1.0, 1: args.pos_weight}, verbose=2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(ckpt, monitor='val_pr_auc', mode='max', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2),
        ],
    ).history
    history = {k: [float(v) for v in vals] for k, vals in history.items()}
    model = keras.models.load_model(ckpt)

    print('[5/7] Değerlendirme ve kalibrasyon...')
    p_val = model.predict(Xva, batch_size=8192, verbose=0).ravel()
    p_test = model.predict(Xte, batch_size=8192, verbose=0).ravel()
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p_val, yva)
    p_test_cal = calibrator.predict(p_test)

    grid = np.linspace(0.02, 0.95, 94)
    best_f1_t = float(grid[int(np.argmax([threshold_metrics(yva, p_val, t)['f1'] for t in grid]))])
    sens80_t = float(np.quantile(p_val[yva == 1], 0.20))

    # Hasta düzeyi: sepsis hastası, etiket başlamadan (ilk pozitif saatten) önce alarm aldı mı?
    patient_detect = []
    for t in (best_f1_t, sens80_t):
        caught, total, fp_patients, neg_total = 0, 0, 0, 0
        for pid in np.unique(gte):
            m = gte == pid
            yp, pp = yte[m], p_test[m]
            if yp.max() == 1:
                total += 1
                first = int(np.argmax(yp == 1))
                caught += int((pp[:first + 1] >= t).any())
            else:
                neg_total += 1
                fp_patients += int((pp >= t).any())
        patient_detect.append({'threshold': t, 'sepsis_patients_alerted': caught / max(total, 1),
                               'non_sepsis_patients_alerted': fp_patients / max(neg_total, 1),
                               'n_sepsis_patients': total, 'n_non_sepsis_patients': neg_total})

    from reliability import mc_dropout_predict
    rng = np.random.default_rng(42)
    sample = np.concatenate([rng.choice(np.where(yte == 1)[0], 150, replace=False),
                             rng.choice(np.where(yte == 0)[0], 150, replace=False)])
    agree, correct = [], []
    for i in sample:
        s = mc_dropout_predict(model, Xte[i:i + 1])
        d = s.mean() >= best_f1_t
        agree.append(np.mean((s >= best_f1_t) == d)); correct.append(d == bool(yte[i]))
    agree, correct = np.array(agree), np.array(correct)
    hi = agree >= 0.9

    print('[6/7] Klinik tutarlılık testi...')
    sanity = clinical_sanity(model, transformer)

    print('[7/7] Kaydediliyor...')
    stats = {
        'model_path': os.path.relpath(ckpt, ROOT).replace('\\', '/'),
        'prevalence_test': float(yte.mean()),
        'n_val': int(len(yva)), 'n_test': int(len(yte)),
        'n_patients': {k: len(v) for k, v in parts.items()},
        'test': {
            'roc_auc': float(roc_auc_score(yte, p_test)),
            'pr_auc': float(average_precision_score(yte, p_test)),
            'brier_raw': float(brier_score_loss(yte, p_test)),
            'brier_calibrated': float(brier_score_loss(yte, p_test_cal)),
            'ece_raw': ece(yte, p_test), 'ece_calibrated': ece(yte, p_test_cal),
            'at_config_threshold': threshold_metrics(yte, p_test, best_f1_t),
            'at_val_best_f1_threshold': threshold_metrics(yte, p_test, best_f1_t),
            'at_val_sens80_threshold': threshold_metrics(yte, p_test, sens80_t),
            'roc_auc_hour_ge_6': float(roc_auc_score(yte[hte >= 5], p_test[hte >= 5])),
        },
        'val_best_f1_threshold': best_f1_t,
        'val_sens80_threshold': sens80_t,
        'patient_level': patient_detect,
        'reliability_curve_raw': reliability_curve(yte, p_test),
        'reliability_curve_calibrated': reliability_curve(yte, p_test_cal),
        'mc_dropout_check': {
            'n': int(len(sample)),
            'accuracy_high_agreement': float(correct[hi].mean()) if hi.any() else None,
            'accuracy_low_agreement': float(correct[~hi].mean()) if (~hi).any() else None,
            'share_low_agreement': float((~hi).mean()),
        },
        'clinical_sanity': sanity,
        'epochs_trained': len(history['loss']),
        'training_minutes': (time.time() - t0) / 60,
    }

    with open(os.path.join(args.output, 'transformer.pkl'), 'wb') as f:
        pickle.dump(transformer, f)
    with open(os.path.join(args.output, 'calibrator.pkl'), 'wb') as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(args.output, 'reliability_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    card = {
        'name': 'Bi-GRU(64) + GRU(64) + Dense',
        'version': 'v3 (ileri doldurma + ölçüm maskeleri)',
        'params': int(model.count_params()),
        'roc_auc': stats['test']['roc_auc'],
        'pr_auc': stats['test']['pr_auc'],
        'best_f1': stats['test']['at_val_best_f1_threshold']['f1'],
        'threshold': best_f1_t,
        'input_shape': [WINDOW, len(feature_names())],
        'loss_function': f'Binary Cross-Entropy (pozitif sınıf ağırlığı {args.pos_weight})',
        'trained_date': time.strftime('%Y-%m-%d'),
        'feature_set': 'v3',
        'architecture': [
            f'Input({WINDOW} saat × {len(feature_names())} özellik: 34 değer + 34 ölçüm maskesi + yaş, cinsiyet, saat)',
            'GaussianNoise(0.1) — yalnızca eğitimde',
            'Bidirectional GRU(64) → 128, L2(1e-4)',
            'Dropout(0.4)',
            'GRU(64), L2(1e-4)',
            'Dropout(0.4)',
            'Dense(32, ReLU) → Dropout(0.4)',
            'Dense(1, Sigmoid) → P(Sepsis)',
        ],
    }
    with open(os.path.join(args.output, 'model_card.json'), 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    save_figures(args.output, history, yte, p_test, p_test_cal)

    print(json.dumps({k: stats[k] for k in ('test', 'patient_level', 'clinical_sanity',
                                             'mc_dropout_check', 'val_best_f1_threshold', 'val_sens80_threshold')},
                     indent=2, ensure_ascii=False))
    print(f'Bitti: {stats["training_minutes"]:.1f} dk')


if __name__ == '__main__':
    main()
