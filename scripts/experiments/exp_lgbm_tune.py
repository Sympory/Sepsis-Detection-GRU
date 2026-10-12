"""
Deney: LightGBM için küçük hiperparametre taraması (yalnızca validasyon).
Az sayıda yapılandırma denenir; çok deneme validasyonu da iyimser yapar.

  py -3.10 scripts/experiments/exp_lgbm_tune.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features_v4 import build_v4

OUT = os.path.join(ROOT, 'models', 'experiments')

BASE = dict(objective='binary', metric='None', learning_rate=0.03, num_leaves=63, min_child_samples=200,
            feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            scale_pos_weight=5.0, verbose=-1, seed=42)
CONFIGS = {
    'base': {},
    'small_trees': dict(num_leaves=31, min_child_samples=500),
    'strong_reg': dict(num_leaves=31, min_child_samples=1000, lambda_l2=20.0, feature_fraction=0.4),
    'slow_deep': dict(learning_rate=0.015, num_leaves=127, min_child_samples=500, feature_fraction=0.4),
    'no_weight': dict(scale_pos_weight=1.0, num_leaves=31, min_child_samples=500),
}


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    X = build_v4(df, group_col='Patient_ID')
    y = df['SepsisLabel'].to_numpy()
    tr, va, te = np.load(os.path.join(OUT, 'split_index.npy'))
    dtrain = lgb.Dataset(X[tr], y[tr], free_raw_data=False)
    dval = lgb.Dataset(X[va], y[va], reference=dtrain, free_raw_data=False)
    pr = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)

    results = {}
    best = None
    for name, cfg in CONFIGS.items():
        params = {**BASE, **cfg}
        model = lgb.train(params, dtrain, num_boost_round=4000, valid_sets=[dval], feval=pr,
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        p_val = model.predict(X[va], num_iteration=model.best_iteration)
        r = {'val_roc_auc': float(roc_auc_score(y[va], p_val)), 'val_pr_auc': float(average_precision_score(y[va], p_val)),
             'best_iteration': int(model.best_iteration), 'params': cfg}
        results[name] = r
        print(f"{name:12s} AUC {r['val_roc_auc']:.4f}  PR {r['val_pr_auc']:.4f}  ağaç {r['best_iteration']}", flush=True)
        if best is None or r['val_pr_auc'] > results[best]['val_pr_auc']:
            best = name
            np.save(os.path.join(OUT, 'lgbm_best_val.npy'), p_val)
            np.save(os.path.join(OUT, 'lgbm_best_test.npy'), model.predict(X[te], num_iteration=model.best_iteration))
            with open(os.path.join(OUT, 'lgbm_best.txt'), 'w', encoding='utf-8') as f:
                f.write(model.model_to_string(num_iteration=model.best_iteration))
    results['_best'] = best
    with open(os.path.join(OUT, 'lgbm_tune.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('En iyi:', best)


if __name__ == '__main__':
    main()
