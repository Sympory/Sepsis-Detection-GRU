"""
models/v4/trees.json dosyasına düğüm değerlerini (açıklanabilirlik için) ekler.

LightGBM aynı veri, aynı ayarlar ve aynı seed'lerle yeniden eğitilir. Yeni ağaçların
tahminleri, uygulamada kullanılan mevcut ağaçlarla BİREBİR aynıysa dosya güncellenir;
değilse hiçbir şey yazılmaz (model değişmemeli).

  py -3.10 scripts/enrich_trees.py
"""

import lightgbm as lgb  # noqa: F401  (diğer kütüphanelerden önce yüklenmeli)

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features_v4 import build_v4
from gbm_numpy import NumpyGBM, export_lightgbm
from train_hybrid import LGBM_PARAMS, SEEDS
from train_v3 import split_patients

TREES = os.path.join(ROOT, 'models', 'v4', 'trees.json')


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    pid = df['Patient_ID'].to_numpy()
    tr, va = (np.isin(pid, list(parts[k])) for k in ('train', 'val'))
    y = df['SepsisLabel'].to_numpy()
    F = build_v4(df, group_col='Patient_ID')

    dtrain = lgb.Dataset(F[tr], y[tr], free_raw_data=False)
    dval = lgb.Dataset(F[va], y[va], reference=dtrain, free_raw_data=False)
    pr = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)
    boosters = []
    for seed in SEEDS:
        params = dict(LGBM_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        b = lgb.train(params, dtrain, num_boost_round=4000, valid_sets=[dval], feval=pr,
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        boosters.append(lgb.Booster(model_str=b.model_to_string(num_iteration=b.best_iteration)))
        print(f'  seed {seed}: {boosters[-1].num_trees()} ağaç', flush=True)

    tmp = TREES + '.yeni'
    export_lightgbm(boosters, tmp)
    old, new = NumpyGBM.load(TREES), NumpyGBM.load(tmp)
    Xv = F[va].to_numpy()[:30000]
    diff = float(np.abs(old.predict(Xv) - new.predict(Xv)).max())
    print(f'Mevcut model ile fark: {diff:.2e}')
    if diff > 1e-9:
        os.remove(tmp)
        sys.exit('Yeniden eğitilen ağaçlar mevcut modelden farklı; dosya DEĞİŞTİRİLMEDİ.')

    # Katkılar + taban = ham skor olmalı (Saabas özdeşliği)
    for i in range(0, 300, 30):
        raw = np.mean([m.predict_raw(Xv[i:i + 1])[0] for m in new.members])
        c, b = new.contributions(Xv[i])
        assert abs(c.sum() + b - raw) < 1e-6, (c.sum() + b, raw)
    shutil.move(tmp, TREES)
    print('trees.json güncellendi: aynı model + düğüm değerleri (açıklanabilirlik)')


if __name__ == '__main__':
    main()
