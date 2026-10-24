"""
Aşama 4 — Birleşim (ensemble): LightGBM (3 seed, Aşama 3 ayarları) + sinir ağı (MLP), aynı 5 katta.

Kat dışı (out-of-fold) tahminler models/v5/oof_*.npy'ye kaydedilir; birleşim yöntemleri bunlar üzerinde
karşılaştırılır. Birleşim ağırlığı DÜRÜST seçilir: her kat için ağırlık, o kat hariç diğer 4 katın tahminlerinden
belirlenir (leave-one-fold-out), sonra o kata uygulanır.

  py -3.10 scripts/v5/birlesim.py            # eğit + değerlendir (önce LightGBM, sonra TensorFlow)
  py -3.10 scripts/v5/birlesim.py --sadece-degerlendir   # kayıtlı tahminlerle yalnızca birleşim
  py -3.10 scripts/v5/birlesim.py --final    # final yapılandırma (monoton kısıtlı) → models/v5/cross_validation.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402  (lightgbm ilk)

import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from deney import DENEYLER  # noqa: E402
from features_v4 import V4Scaler, build_v4  # noqa: E402

BEST = DENEYLER['ayar_d12'][2]
SEEDS = (42, 7, 2024)
OOF = {name: os.path.join(lab.OUT_DIR, f'oof_{name}.npy') for name in ('gbm3', 'gbm3m', 'mlp')}
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def fold_metrics(p, y, fold):
    return [{'roc_auc': float(roc_auc_score(y[fold == k], p[fold == k])),
             'pr_auc': float(average_precision_score(y[fold == k], p[fold == k]))} for k in range(lab.K)]


def train_mlp_oof(F, y, fold, inner):
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from train_hybrid import build_mlp
    cols = [str(i) for i in range(F.shape[1])]
    oof = np.zeros(len(y))
    for k in range(lab.K):
        tf.keras.utils.set_random_seed(42 + k)
        te, tr, va = fold == k, (fold != k) & ~inner, (fold != k) & inner
        scaler = V4Scaler().fit(pd.DataFrame(F[tr], columns=cols))
        X = lambda m: scaler.transform(pd.DataFrame(F[m], columns=cols))
        mlp = build_mlp(F.shape[1])
        mlp.fit(X(tr), y[tr], validation_data=(X(va), y[va]), epochs=30, batch_size=4096,
                class_weight={0: 1.0, 1: 5.0}, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=4, restore_best_weights=True),
                           keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)])
        oof[te] = mlp.predict(X(te), batch_size=16384, verbose=0).ravel()
        lab.log(f'  MLP kat {k + 1}: ROC-AUC {roc_auc_score(y[te], oof[te]):.4f}')
    return oof


def rank(p):
    return np.argsort(np.argsort(p)) / len(p)


def final_cv():
    """Final yapılandırma: monoton kısıtlı LightGBM (3 seed) + kayıtlı MLP tahminleri, ağırlık kat dışında seçilir."""
    from deney import DENEYLER as D, monotone_vector
    t0 = time.time()
    dev = lab.load_dev()
    fold, inner = lab.fold_assignments(dev)
    y = dev['SepsisLabel'].to_numpy()
    F, cols = lab.features(dev, 'v4', lambda d: build_v4(d, group_col='Patient_ID'))
    mono = monotone_vector(cols, D['monoton'][2]['monoton'])
    gbm, folds = lab.lgbm_cv(F, y, fold, inner, params=dict(BEST), seeds=SEEDS, monotone=mono)
    np.save(OOF['gbm3m'], gbm)
    mlp = np.load(OOF['mlp'])
    blend, chosen = np.zeros(len(y)), []
    for k in range(lab.K):
        other = fold != k
        w = float(WEIGHTS[int(np.argmax([roc_auc_score(y[other], w * gbm[other] + (1 - w) * mlp[other]) for w in WEIGHTS]))])
        chosen.append(w)
        blend[fold == k] = w * gbm[fold == k] + (1 - w) * mlp[fold == k]
    lab.save_result('final_v5', 'Final: monoton LightGBM 3 seed + MLP, ağırlık kat dışında', fold_metrics(blend, y, fold))
    res = {'k': lab.K, 'n_patients': int(dev['Patient_ID'].nunique()), 'note': 'Geliştirme hastaları (kilitli test hariç)',
           'weights_lightgbm': chosen, 'folds': [], 'summary': {}}
    preds = (('hybrid', blend), ('lightgbm', gbm), ('mlp', mlp))
    for k in range(lab.K):
        m = fold == k
        res['folds'].append({n: {'roc_auc': float(roc_auc_score(y[m], p[m])),
                                 'pr_auc': float(average_precision_score(y[m], p[m]))} for n, p in preds})
    for n, _ in preds:
        for met in ('roc_auc', 'pr_auc'):
            v = np.array([f[n][met] for f in res['folds']])
            res['summary'][f'{n}_{met}'] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)),
                                            'min': float(v.min()), 'max': float(v.max())}
    res['minutes'] = (time.time() - t0) / 60
    import json
    with open(os.path.join(lab.OUT_DIR, 'cross_validation.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for n, _ in preds:
        a = res['summary'][f'{n}_roc_auc']
        lab.log(f"{n:9s} ROC-AUC {a['mean']:.4f} ± {a['std']:.4f}")
    lab.log(lab.compare('final_v5')[0])


def main():
    if '--final' in sys.argv:
        return final_cv()
    t0 = time.time()
    dev = lab.load_dev()
    fold, inner = lab.fold_assignments(dev)
    y = dev['SepsisLabel'].to_numpy()

    if '--sadece-degerlendir' not in sys.argv:
        F, _ = lab.features(dev, 'v4', lambda d: build_v4(d, group_col='Patient_ID'))
        lab.log('LightGBM 3 seed (Aşama 3 ayarları)…')
        gbm, folds = lab.lgbm_cv(F, y, fold, inner, params=dict(BEST), seeds=SEEDS)
        np.save(OOF['gbm3'], gbm)
        lab.save_result('gbm3_d12', 'LightGBM 3 seed, Aşama 3 ayarları', folds)
        lab.log(f'  {(time.time() - t0) / 60:.1f} dk · sinir ağı…')
        mlp = train_mlp_oof(F, y, fold, inner)
        np.save(OOF['mlp'], mlp)
        lab.save_result('mlp', 'Sinir ağı (MLP), v4 özellikleri', fold_metrics(mlp, y, fold))

    gbm, mlp = np.load(OOF['gbm3']), np.load(OOF['mlp'])
    rows = {}

    # Sabit 0.7/0.3 (mevcut uygulamadaki gibi)
    rows['hibrit_sabit'] = (0.7 * gbm + 0.3 * mlp, 'LightGBM×0.7 + MLP×0.3 (sabit ağırlık)')

    # Ağırlık, kat dışı seçilir: k. kat için diğer 4 katta en iyi ağırlık
    blend, chosen = np.zeros(len(y)), []
    for k in range(lab.K):
        other = fold != k
        scores = [roc_auc_score(y[other], w * gbm[other] + (1 - w) * mlp[other]) for w in WEIGHTS]
        w = float(WEIGHTS[int(np.argmax(scores))])
        chosen.append(w)
        blend[fold == k] = w * gbm[fold == k] + (1 - w) * mlp[fold == k]
    rows['hibrit_secilen'] = (blend, f'Ağırlık kat dışında seçildi (LightGBM payı: {chosen})')

    # Sıra ortalaması: iki modelin olasılık ölçekleri farklıysa daha dengeli
    rk = np.zeros(len(y))
    for k in range(lab.K):
        m = fold == k
        rk[m] = 0.5 * rank(gbm[m]) + 0.5 * rank(mlp[m])
    rows['hibrit_sira'] = (rk, 'Sıra (rank) ortalaması, eşit ağırlık')

    for name, (p, desc) in rows.items():
        lab.save_result(name, desc, fold_metrics(p, y, fold))

    lab.log('\nKarşılaştırma (temel çizgi = tek seed LightGBM, mevcut ayarlar):')
    for name in ('ayar_d12', 'gbm3_d12', 'mlp', 'hibrit_sabit', 'hibrit_secilen', 'hibrit_sira'):
        s = lab.summarize(lab.load_log()[name]['folds'])
        lab.log(f'\n{name:15s} ROC-AUC {s["roc_auc"]["mean"]:.4f} ± {s["roc_auc"]["std"]:.4f} · PR-AUC {s["pr_auc"]["mean"]:.4f}')
        cmp = lab.compare(name)
        if cmp:
            lab.log(cmp[0])
    lab.log(f'\n{(time.time() - t0) / 60:.1f} dk')


if __name__ == '__main__':
    main()
