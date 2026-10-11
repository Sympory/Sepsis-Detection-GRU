"""
GRU v3 ile aynı hasta bölmesi ve aynı özellikler üzerinde LightGBM karşılaştırması.

LightGBM ile TensorFlow aynı süreçte Windows'ta DLL çakışması yaşadığı için
bu betik TensorFlow'u hiç yüklemez. Sonuç models/v3/reliability_stats.json
içine 'lightgbm_baseline' olarak eklenir.

Kullanım (train_v3.py'den sonra):
  python scripts/baseline_lightgbm.py
"""

import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import CLINICAL_COLUMNS, build_hourly_features
from train_v3 import split_patients, windows_for

MODEL_DIR = os.path.join(ROOT, 'models', 'v3')


def flatten(X):
    """Son saat + pencere içindeki değişim (son - ilk) → tablo formu"""
    n = len(CLINICAL_COLUMNS)
    return np.concatenate([X[:, -1, :], X[:, -1, :n] - X[:, 0, :n]], axis=1)


def main():
    with open(os.path.join(MODEL_DIR, 'transformer.pkl'), 'rb') as f:
        transformer = pickle.load(f)

    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    X_all = transformer.transform(build_hourly_features(df, group_col='Patient_ID'))

    Xtr, ytr, _, _ = windows_for(df, X_all, parts['train'])
    Xte, yte, _, _ = windows_for(df, X_all, parts['test'])
    del X_all

    rng = np.random.default_rng(42)
    idx = rng.choice(len(ytr), min(len(ytr), 500_000), replace=False)
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)
    clf.fit(flatten(Xtr[idx]), ytr[idx])
    p = clf.predict_proba(flatten(Xte))[:, 1]
    result = {'roc_auc': float(roc_auc_score(yte, p)), 'pr_auc': float(average_precision_score(yte, p)),
              'train_samples': int(len(idx))}

    stats_path = os.path.join(MODEL_DIR, 'reliability_stats.json')
    with open(stats_path, encoding='utf-8') as f:
        stats = json.load(f)
    stats['lightgbm_baseline'] = result
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
