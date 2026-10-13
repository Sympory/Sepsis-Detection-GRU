"""
Hibrit modelin 5 katlı, hasta bazlı çapraz doğrulaması (GroupKFold, sepsis durumuna göre tabakalı).

Her katta: o kattaki hastalar test; kalan hastaların %80'i eğitim, %20'si erken durdurma
için iç validasyon. Ayarlar ana modelle aynı (LightGBM tek seed, MLP, birleşim 0.7/0.3).
Sonuç: her model için ROC-AUC ve PR-AUC'nin ortalaması ± standart sapması.

İki aşama: önce tüm katlarda LightGBM (TensorFlow yüklenmeden; Windows'ta DLL çakışması),
sonra sinir ağı.

  py -3.10 scripts/cv_hybrid.py      → models/v4/cross_validation.json
"""

import lightgbm as lgb  # noqa: F401  (diğer kütüphanelerden önce yüklenmeli)

import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features_v4 import V4Scaler, build_v4
from train_hybrid import LGBM_PARAMS

K = 5
W_GBM = 0.7
OUT = os.path.join(ROOT, 'models', 'v4', 'cross_validation.json')


def fold_assignments(df, seed=42):
    """Hastaları sepsis durumuna göre tabakalı olarak K kata dağıtır; ayrıca iç validasyon işareti."""
    status = df.groupby('Patient_ID')['SepsisLabel'].max()
    rng = np.random.default_rng(seed)
    fold, inner_val = {}, {}
    for label in (0, 1):
        ids = status[status == label].index.to_numpy()
        rng.shuffle(ids)
        for i, p in enumerate(ids):
            fold[p] = i % K
            inner_val[p] = rng.random() < 0.2
    return fold, inner_val


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    fold, inner = fold_assignments(df)
    f = df['Patient_ID'].map(fold).to_numpy()
    iv = df['Patient_ID'].map(inner).to_numpy()
    y = df['SepsisLabel'].to_numpy()
    F = build_v4(df, group_col='Patient_ID')
    print(f'{F.shape[1]} özellik · {(time.time() - t0) / 60:.1f} dk', flush=True)

    pr = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)
    p_gbm = np.zeros(len(df))
    for k in range(K):
        te, tr, va = f == k, (f != k) & ~iv, (f != k) & iv
        dtrain = lgb.Dataset(F[tr], y[tr], free_raw_data=False)
        dval = lgb.Dataset(F[va], y[va], reference=dtrain, free_raw_data=False)
        b = lgb.train(dict(LGBM_PARAMS, seed=42, bagging_seed=42, feature_fraction_seed=42), dtrain,
                      num_boost_round=4000, valid_sets=[dval], feval=pr,
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p_gbm[te] = b.predict(F[te], num_iteration=b.best_iteration)
        print(f'  LightGBM kat {k + 1}: AUC {roc_auc_score(y[te], p_gbm[te]):.4f} · {(time.time() - t0) / 60:.1f} dk', flush=True)

    import tensorflow as tf
    from tensorflow import keras
    from train_hybrid import build_mlp
    p_mlp = np.zeros(len(df))
    for k in range(K):
        tf.keras.utils.set_random_seed(42 + k)
        te, tr, va = f == k, (f != k) & ~iv, (f != k) & iv
        scaler = V4Scaler().fit(F[tr])
        X = scaler.transform(F)
        mlp = build_mlp(X.shape[1])
        mlp.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=30, batch_size=4096,
                class_weight={0: 1.0, 1: 5.0}, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=4, restore_best_weights=True),
                           keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)])
        p_mlp[te] = mlp.predict(X[te], batch_size=16384, verbose=0).ravel()
        del X
        print(f'  Sinir ağı kat {k + 1}: AUC {roc_auc_score(y[te], p_mlp[te]):.4f} · {(time.time() - t0) / 60:.1f} dk', flush=True)

    p_hyb = W_GBM * p_gbm + (1 - W_GBM) * p_mlp
    res = {'k': K, 'n_patients': int(df['Patient_ID'].nunique()), 'folds': [], 'summary': {}}
    for k in range(K):
        te = f == k
        res['folds'].append({name: {'roc_auc': float(roc_auc_score(y[te], p[te])),
                                    'pr_auc': float(average_precision_score(y[te], p[te]))}
                             for name, p in (('hybrid', p_hyb), ('lightgbm', p_gbm), ('mlp', p_mlp))})
    for name in ('hybrid', 'lightgbm', 'mlp'):
        for m in ('roc_auc', 'pr_auc'):
            v = np.array([fd[name][m] for fd in res['folds']])
            res['summary'][f'{name}_{m}'] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)),
                                             'min': float(v.min()), 'max': float(v.max())}
    res['minutes'] = (time.time() - t0) / 60
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    for name in ('hybrid', 'lightgbm', 'mlp'):
        a, p = res['summary'][f'{name}_roc_auc'], res['summary'][f'{name}_pr_auc']
        print(f"{name:9s} ROC-AUC {a['mean']:.4f} ± {a['std']:.4f}   PR-AUC {p['mean']:.4f} ± {p['std']:.4f}")


if __name__ == '__main__':
    main()
