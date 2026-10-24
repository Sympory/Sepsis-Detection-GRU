"""
Sepsis Hibrit Model (v4 / v5) — eğitim, kalibrasyon ve TEK SEFERLİK test değerlendirmesi
=======================================================================================

İki profil (SEPSIS_SURUM ortam değişkeni, varsayılan v5):
  v4 — ilk hibrit: LightGBM öğrenme hızı 0.03, 31 yaprak; birleşim ağırlığı PR-AUC ile seçilir.
  v5 — model geliştirme deneylerinin sonucu (scripts/v5, models/v5/deneyler.json): ayar arama #12,
       MAP/SBP/SpO2/laktat için monoton kısıt, birleşim ağırlığı ROC-AUC ile seçilir.

Model: zaman serisi özellikleri (features_v4, 281 özellik) üzerinde
    LightGBM (3 seed topluluğu)  +  çok katmanlı sinir ağı (MLP)
olasılıkların ağırlıklı ortalaması → izotonik kalibrasyon.

Tüm seçimler (erken durdurma, birleşim ağırlığı, kalibrasyon, eşik) VALİDASYON
hastalarıyla yapılır. Test hastaları yalnızca en sonda bir kez puanlanır.

Sıra bilinçli: LightGBM eğitimi TensorFlow yüklenmeden yapılır (Windows'ta aynı
süreçte DLL çakışması var). Ağaçlar JSON'a aktarılır ve uygulamada saf NumPy ile
çalıştırılır (gbm_numpy.py); NumPy tahmininin LightGBM ile aynı olduğu doğrulanır.

Çıktılar: models/<sürüm>/
  trees.json, mlp.keras, scaler.pkl, calibrator.pkl, model_card.json,
  reliability_stats.json, training_history.json, evaluation.png

  py -3.10 scripts/train_hybrid.py                    # v5
  set SEPSIS_SURUM=v4 && py -3.10 scripts/train_hybrid.py   # ilk hibrit model (yeniden üretmek için)
"""

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

# LightGBM diğer kütüphanelerden (sklearn/scipy/TensorFlow) ÖNCE yüklenmeli: Windows'ta
# OpenMP kütüphaneleri çakışırsa LightGBM eğitimi 'access violation' ile çöküyor.
import lightgbm as lgb  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from features import CLINICAL_COLUMNS
from features_v4 import V4Scaler, build_v4
from fit_calibration import ece, reliability_curve, threshold_metrics
from gbm_numpy import NumpyGBM, export_lightgbm
from train_v3 import split_patients

VERSION = os.environ.get('SEPSIS_SURUM', 'v5')
OUT = os.path.join(ROOT, 'models', VERSION)
SEEDS = (42, 7, 2024)
_V4_LGBM = dict(objective='binary', metric='None', learning_rate=0.03, num_leaves=31, min_child_samples=1000,
                feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0,
                scale_pos_weight=5.0, verbose=-1)
# v5: scripts/v5/ayar_ara.py deneme #12, 5 katta doğrulandı (ROC-AUC +0.0022, 5/5 kat)
_V5_LGBM = dict(_V4_LGBM, learning_rate=0.0106, num_leaves=32, min_child_samples=2274, feature_fraction=0.209,
                bagging_fraction=0.669, lambda_l2=4.40, scale_pos_weight=1.87, max_depth=12, min_gain_to_split=0.1)
LGBM_PARAMS = _V5_LGBM if VERSION == 'v5' else _V4_LGBM
# v5: değer kötüleştikçe risk düşemez (−1: değer düştükçe risk artar). PR-AUC 5/5 katta +, ROC-AUC nötr
MONOTONE = {'MAP': -1, 'SBP': -1, 'O2Sat': -1, 'Lactate': 1} if VERSION == 'v5' else {}
BLEND_METRIC = roc_auc_score if VERSION == 'v5' else average_precision_score


def monotone_vector(cols):
    """Değişkenin kendisi ve pencere istatistikleri (ortalama/min/maks) kısıtlanır; süre, sayı, değişim serbest."""
    vec = []
    for c in cols:
        base, _, suffix = c.partition('_')
        ok = suffix == '' or any(suffix.startswith(x) for x in ('mean', 'min', 'max'))
        vec.append(MONOTONE.get(base, 0) if ok else 0)
    return vec


def log(msg):
    print(msg, flush=True)


# ============================================================================
# 1) LIGHTGBM (TensorFlow yüklenmeden)
# ============================================================================

def train_lightgbm(F, y, tr, va):
    dtrain = lgb.Dataset(F[tr], y[tr], free_raw_data=False)
    dval = lgb.Dataset(F[va], y[va], reference=dtrain, free_raw_data=False)
    pr = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)
    boosters, info = [], []
    base = dict(LGBM_PARAMS)
    if MONOTONE:
        base['monotone_constraints'] = monotone_vector(list(F.columns))
    for seed in SEEDS:
        params = dict(base, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        b = lgb.train(params, dtrain, num_boost_round=6000, valid_sets=[dval], feval=pr,
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        b = lgb.Booster(model_str=b.model_to_string(num_iteration=b.best_iteration))
        p = b.predict(F[va])
        info.append({'seed': seed, 'trees': b.num_trees(), 'val_roc_auc': float(roc_auc_score(y[va], p)),
                     'val_pr_auc': float(average_precision_score(y[va], p))})
        log(f"  LightGBM seed {seed}: {b.num_trees()} ağaç · val AUC {info[-1]['val_roc_auc']:.4f} PR {info[-1]['val_pr_auc']:.4f}")
        boosters.append(b)
    export_lightgbm(boosters, os.path.join(OUT, 'trees.json'))

    # NumPy değerlendiricisi LightGBM ile aynı sonucu vermeli
    gbm = NumpyGBM.load(os.path.join(OUT, 'trees.json'))
    sample = np.where(va)[0][:20000]
    ref = np.mean([b.predict(F.iloc[sample]) for b in boosters], axis=0)
    diff = float(np.abs(gbm.predict(F.iloc[sample].to_numpy()) - ref).max())
    log(f'  NumPy değerlendiricisi farkı: {diff:.2e}')
    assert diff < 1e-9, 'NumPy ağaç değerlendiricisi LightGBM ile eşleşmiyor'
    imp = pd.Series(np.mean([b.feature_importance('gain') for b in boosters], axis=0),
                    index=boosters[0].feature_name()).sort_values(ascending=False)
    return info, diff, imp


# ============================================================================
# 2) SİNİR AĞI (MLP)
# ============================================================================

def build_mlp(n_features):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    l2 = regularizers.l2(1e-4)
    inp = keras.Input(shape=(n_features,))
    x = layers.GaussianNoise(0.1, name='noise')(inp)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2, name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.5, name='drop_1')(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2, name='dense_128')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.5, name='drop_2')(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2, name='dense_64')(x)
    x = layers.Dropout(0.5, name='drop_3')(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inp, out, name='sepsis_mlp_v4')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='roc_auc'), keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model


# ============================================================================
# 3) DEĞERLENDİRME YARDIMCILARI
# ============================================================================

def patient_level(y, p, pid, t):
    caught, total, fp, neg = 0, 0, 0, 0
    for g in np.unique(pid):
        m = pid == g
        yy, pp = y[m], p[m]
        if yy.max() == 1:
            total += 1
            first = int(np.argmax(yy == 1))
            caught += int((pp[:first + 1] >= t).any())
        else:
            neg += 1
            fp += int((pp >= t).any())
    return {'threshold': float(t), 'sepsis_patients_alerted': caught / max(total, 1),
            'non_sepsis_patients_alerted': fp / max(neg, 1), 'n_sepsis_patients': total, 'n_non_sepsis_patients': neg}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    log('[1/6] Veri ve özellikler...')
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    pid = df['Patient_ID'].to_numpy()
    tr, va, te = (np.isin(pid, list(parts[k])) for k in ('train', 'val', 'test'))
    y = df['SepsisLabel'].to_numpy()
    F = build_v4(df, group_col='Patient_ID')
    log(f'  {len(df):,} saat · {F.shape[1]} özellik · {tr.sum():,}/{va.sum():,}/{te.sum():,} (eğitim/val/test)')

    log('[2/6] LightGBM (3 seed)...')
    lgbm_info, numpy_diff, importance = train_lightgbm(F, y, tr, va)

    log('[3/6] Sinir ağı (MLP)...')
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)
    scaler = V4Scaler().fit(F[tr])
    X = scaler.transform(F)
    mlp = build_mlp(X.shape[1])
    ckpt = os.path.join(OUT, 'mlp.keras')
    history = mlp.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=30, batch_size=4096,
                      class_weight={0: 1.0, 1: 5.0}, verbose=2,
                      callbacks=[keras.callbacks.ModelCheckpoint(ckpt, monitor='val_pr_auc', mode='max', save_best_only=True),
                                 keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=4, restore_best_weights=True),
                                 keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)]).history
    history = {k: [float(v) for v in vals] for k, vals in history.items()}
    mlp = keras.models.load_model(ckpt)

    log('[4/6] Birleşim ağırlığı ve kalibrasyon (validasyon)...')
    gbm = NumpyGBM.load(os.path.join(OUT, 'trees.json'))
    Fv, Ft = F[va].to_numpy(), F[te].to_numpy()
    pg_val, pg_test = gbm.predict(Fv), gbm.predict(Ft)
    pm_val = mlp.predict(X[va], batch_size=16384, verbose=0).ravel()
    pm_test = mlp.predict(X[te], batch_size=16384, verbose=0).ravel()
    weights = np.round(np.arange(0, 1.01, 0.05), 2)
    scores = [BLEND_METRIC(y[va], w * pg_val + (1 - w) * pm_val) for w in weights]
    w_gbm = float(weights[int(np.argmax(scores))])
    p_val = w_gbm * pg_val + (1 - w_gbm) * pm_val
    p_test = w_gbm * pg_test + (1 - w_gbm) * pm_test
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p_val, y[va])
    grid = np.linspace(0.02, 0.95, 94)
    best_f1_t = float(grid[int(np.argmax([threshold_metrics(y[va], p_val, t)['f1'] for t in grid]))])
    sens80_t = float(np.quantile(p_val[y[va] == 1], 0.20))
    log(f'  LightGBM ağırlığı {w_gbm} · MLP {1 - w_gbm:.2f} · F1 eşiği {best_f1_t:.3f} · tarama eşiği {sens80_t:.3f}')

    log('[5/6] TEST (tek seferlik)...')
    yt = y[te]
    p_test_cal = calibrator.predict(p_test)
    comp = {
        'hybrid': {'roc_auc': float(roc_auc_score(yt, p_test)), 'pr_auc': float(average_precision_score(yt, p_test))},
        'lightgbm': {'roc_auc': float(roc_auc_score(yt, pg_test)), 'pr_auc': float(average_precision_score(yt, pg_test))},
        'mlp': {'roc_auc': float(roc_auc_score(yt, pm_test)), 'pr_auc': float(average_precision_score(yt, pm_test))},
    }
    v3_stats = json.load(open(os.path.join(ROOT, 'models', 'v3', 'reliability_stats.json'), encoding='utf-8'))
    comp['gru_v3'] = {'roc_auc': v3_stats['test']['roc_auc'], 'pr_auc': v3_stats['test']['pr_auc']}
    v4_path = os.path.join(ROOT, 'models', 'v4', 'reliability_stats.json')
    if VERSION == 'v5' and os.path.exists(v4_path):
        v4_stats = json.load(open(v4_path, encoding='utf-8'))
        comp['hybrid_v4'] = {'roc_auc': v4_stats['test']['roc_auc'], 'pr_auc': v4_stats['test']['pr_auc']}
    hours = df['Hour'].to_numpy()[te]

    # MC-Dropout (MLP kısmı): belirsizlik, karar uyumu ile doğruluk ilişkisi
    from reliability import mc_dropout_predict
    rng = np.random.default_rng(42)
    te_idx = np.where(te)[0]
    sample = np.concatenate([rng.choice(np.where(yt == 1)[0], 150, replace=False),
                             rng.choice(np.where(yt == 0)[0], 150, replace=False)])
    agree, correct = [], []
    for i in sample:
        s = w_gbm * pg_test[i] + (1 - w_gbm) * mc_dropout_predict(mlp, X[te_idx[i]:te_idx[i] + 1])
        d = s.mean() >= best_f1_t
        agree.append(np.mean((s >= best_f1_t) == d))
        correct.append(d == bool(yt[i]))
    agree, correct = np.array(agree), np.array(correct)
    hi = agree >= 0.9

    log('[6/6] Klinik tutarlılık ve kayıt...')

    def predict_rows(rows):
        f = build_v4(pd.DataFrame(rows))
        return float(w_gbm * gbm.predict(f.iloc[[-1]].to_numpy())[0]
                     + (1 - w_gbm) * mlp.predict(scaler.transform(f.iloc[[-1]]), verbose=0)[0, 0])

    normal = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75, 'MAP': 92,
              'Lactate': 1.2, 'WBC': 8, 'Age': 65, 'Gender': 1}
    septic = dict(normal, HR=130, O2Sat=88, Temp=38.9, Resp=28, SBP=82, DBP=48, MAP=59, Lactate=4.5, WBC=17)

    def scenario(values, hour, change_last=6, base=normal):
        return [dict(values if h > hour - change_last else base, Hour=h) for h in range(hour + 1)]

    sanity = {'normal_vs_septic': {}, 'single_feature': {}}
    for hour in (6, 24, 72):
        sanity['normal_vs_septic'][hour] = {'normal': predict_rows(scenario(normal, hour)),
                                            'septic': predict_rows(scenario(septic, hour))}
    base = predict_rows(scenario(normal, 24))
    for feat, bad in (('HR', 135), ('Temp', 39.2), ('Resp', 30), ('MAP', 58), ('SBP', 80),
                      ('O2Sat', 87), ('Lactate', 5.0), ('WBC', 20)):
        sanity['single_feature'][feat] = {'value': bad, 'delta': predict_rows(scenario(dict(normal, **{feat: bad}), 24)) - base}
    sanity['passed'] = (all(v['septic'] > v['normal'] for v in sanity['normal_vs_septic'].values())
                        and sum(v['delta'] >= 0 for v in sanity['single_feature'].values()) >= 6)

    stats = {
        'model_path': f'models/{VERSION}',
        'prevalence_test': float(yt.mean()), 'n_val': int(va.sum()), 'n_test': int(te.sum()),
        'n_patients': {k: len(v) for k, v in parts.items()},
        'test': {
            'roc_auc': comp['hybrid']['roc_auc'], 'pr_auc': comp['hybrid']['pr_auc'],
            'brier_raw': float(brier_score_loss(yt, p_test)), 'brier_calibrated': float(brier_score_loss(yt, p_test_cal)),
            'ece_raw': ece(yt, p_test), 'ece_calibrated': ece(yt, p_test_cal),
            'at_config_threshold': threshold_metrics(yt, p_test, best_f1_t),
            'at_val_best_f1_threshold': threshold_metrics(yt, p_test, best_f1_t),
            'at_val_sens80_threshold': threshold_metrics(yt, p_test, sens80_t),
            'roc_auc_hour_ge_6': float(roc_auc_score(yt[hours >= 5], p_test[hours >= 5])),
        },
        'components_test': comp,
        'val_best_f1_threshold': best_f1_t, 'val_sens80_threshold': sens80_t,
        'ensemble_weight_lightgbm': w_gbm,
        'patient_level': [patient_level(yt, p_test, pid[te], t) for t in (best_f1_t, sens80_t)],
        'reliability_curve_raw': reliability_curve(yt, p_test),
        'reliability_curve_calibrated': reliability_curve(yt, p_test_cal),
        'mc_dropout_check': {'n': int(len(sample)),
                             'accuracy_high_agreement': float(correct[hi].mean()) if hi.any() else None,
                             'accuracy_low_agreement': float(correct[~hi].mean()) if (~hi).any() else None,
                             'share_low_agreement': float((~hi).mean())},
        'clinical_sanity': sanity,
        'lightgbm_seeds': lgbm_info, 'numpy_vs_lightgbm_max_diff': numpy_diff,
        'top_features': [[k, round(float(v), 1)] for k, v in importance.head(20).items()],
        'training_minutes': (time.time() - t0) / 60,
    }
    card = {
        'name': 'Hibrit: LightGBM (3 seed) + Sinir Ağı (MLP)',
        'version': 'v5 (ayarlı, monoton kısıtlı)' if VERSION == 'v5' else 'v4 (zaman serisi özellikleri)',
        'feature_set': 'v4',
        'params': int(mlp.count_params()),
        'n_trees': int(sum(i['trees'] for i in lgbm_info)),
        'roc_auc': stats['test']['roc_auc'], 'pr_auc': stats['test']['pr_auc'],
        'best_f1': stats['test']['at_val_best_f1_threshold']['f1'],
        'threshold': best_f1_t,
        'ensemble_weight_lightgbm': w_gbm,
        'input_shape': [int(X.shape[1])],
        'loss_function': f'Binary Cross-Entropy (pozitif sınıf ağırlığı 5) · LightGBM: binary, scale_pos_weight {LGBM_PARAMS["scale_pos_weight"]}',
        'trained_date': time.strftime('%Y-%m-%d'),
        'architecture': [
            f'Girdi: {X.shape[1]} zaman serisi özelliği (son değer, ölçüm maskesi, son ölçümden geçen süre, '
            '6/24 saatlik ve yatıştan bu yana ortalama-min-maks-std, değişim, SIRS/SOFA/şok indeksi)',
            f'Dal 1 — LightGBM: 3 seed × ~{int(np.mean([i["trees"] for i in lgbm_info]))} ağaç, '
            f'{LGBM_PARAMS["num_leaves"]} yaprak, öğrenme hızı {LGBM_PARAMS["learning_rate"]}, L2={LGBM_PARAMS["lambda_l2"]}'
            + (', monoton kısıt: MAP/SBP/SpO2 ↓ veya laktat ↑ iken risk düşemez' if MONOTONE else '')
            + ' (uygulamada saf NumPy ile çalışır)',
            'Dal 2 — Sinir ağı: GaussianNoise(0.1) → Dense(256, ReLU) → BatchNorm → Dropout(0.5) '
            '→ Dense(128, ReLU) → BatchNorm → Dropout(0.5) → Dense(64, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)',
            f'Birleşim: {w_gbm:.2f} × LightGBM + {1 - w_gbm:.2f} × MLP (validasyonda seçildi)',
            'İzotonik kalibrasyon → P(Sepsis)',
        ],
    }

    with open(os.path.join(OUT, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT, 'calibrator.pkl'), 'wb') as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(OUT, 'reliability_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, 'model_card.json'), 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    from train_v3 import save_figures
    save_figures(OUT, history, yt, p_test, p_test_cal)

    log(json.dumps({'components_test': comp, 'weight_lgbm': w_gbm,
                    'at_f1': {k: round(v, 4) if isinstance(v, float) else v for k, v in stats['test']['at_val_best_f1_threshold'].items()},
                    'at_sens80': {k: round(v, 4) if isinstance(v, float) else v for k, v in stats['test']['at_val_sens80_threshold'].items()},
                    'patient_level': stats['patient_level'], 'sanity': sanity, 'mc': stats['mc_dropout_check'],
                    'ece': [stats['test']['ece_raw'], stats['test']['ece_calibrated']]}, indent=1, ensure_ascii=False))
    log(f'Bitti: {stats["training_minutes"]:.1f} dk')


if __name__ == '__main__':
    main()
