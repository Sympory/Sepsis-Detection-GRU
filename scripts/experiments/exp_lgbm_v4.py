"""
Deney: v4 zaman serisi özellikleriyle LightGBM.

Seçim yalnızca VALİDASYON üzerinde yapılır; test tahminleri kaydedilir ama
burada puanlanmaz (son değerlendirme scripts/experiments/final_eval.py'de bir kez).

  py -3.10 scripts/experiments/exp_lgbm_v4.py
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features_v4 import build_v4
from train_v3 import split_patients

OUT = os.path.join(ROOT, 'models', 'experiments')


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    print(f'Veri: {len(df):,} saat · özellikler hesaplanıyor...', flush=True)
    X = build_v4(df, group_col='Patient_ID')
    print(f'  {X.shape[1]} özellik · {(time.time() - t0) / 60:.1f} dk', flush=True)
    y = df['SepsisLabel'].to_numpy()
    pid = df['Patient_ID'].to_numpy()

    idx = {k: np.isin(pid, list(v)) for k, v in parts.items()}
    np.save(os.path.join(OUT, 'split_index.npy'), np.stack([idx['train'], idx['val'], idx['test']]))

    # metric='None': erken durdurma yalnızca PR-AUC'ye bakar. (Varsayılan log-loss, sınıf ağırlığı yüzünden
    # eğitim ilerledikçe kötüleşir ve modeli 7. ağaçta durduruyordu.)
    params = dict(objective='binary', metric='None', learning_rate=0.03, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  scale_pos_weight=5.0, verbose=-1, seed=42)
    dtrain = lgb.Dataset(X[idx['train']], y[idx['train']])
    dval = lgb.Dataset(X[idx['val']], y[idx['val']], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      feval=lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True),
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(200)])

    p_val = model.predict(X[idx['val']], num_iteration=model.best_iteration)
    p_test = model.predict(X[idx['test']], num_iteration=model.best_iteration)
    np.save(os.path.join(OUT, 'lgbm_v4_val.npy'), p_val)
    np.save(os.path.join(OUT, 'lgbm_v4_test.npy'), p_test)
    # LightGBM'in C kütüphanesi Türkçe karakterli yollara yazamıyor; metni Python ile yaz
    with open(os.path.join(OUT, 'lgbm_v4.txt'), 'w', encoding='utf-8') as f:
        f.write(model.model_to_string(num_iteration=model.best_iteration))

    imp = pd.Series(model.feature_importance('gain'), index=X.columns).sort_values(ascending=False)
    res = {'val_roc_auc': float(roc_auc_score(y[idx['val']], p_val)),
           'val_pr_auc': float(average_precision_score(y[idx['val']], p_val)),
           'best_iteration': int(model.best_iteration), 'n_features': int(X.shape[1]),
           'top_features': imp.head(25).round(0).to_dict(), 'minutes': (time.time() - t0) / 60}
    with open(os.path.join(OUT, 'lgbm_v4.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'top_features'}, indent=2))
    print('En önemli 15:', list(imp.head(15).index))


if __name__ == '__main__':
    main()
