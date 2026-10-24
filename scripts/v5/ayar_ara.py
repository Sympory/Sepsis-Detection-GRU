"""
LightGBM için rastgele ayar arama (ek paket gerektirmez).

Her denemede ayarlar belirli aralıklardan rastgele seçilir; hız için 5 katın 3'ünde (kat 1–3) ölçülür.
En iyi ayarlar daha sonra deney.py ile TAM 5 katta, karar kuralıyla doğrulanır — aramada kazanan ayar
şans eseri iyi çıkmış olabilir, doğrulama bunu ayıklar.

  py -3.10 scripts/v5/ayar_ara.py 30          # 30 deneme (her biri ~2–4 dk)
Sonuçlar: models/v5/ayar_arama.json (kaldığı yerden devam eder)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from features_v4 import build_v4  # noqa: E402

PATH = os.path.join(lab.OUT_DIR, 'ayar_arama.json')
SEARCH_FOLDS = [0, 1, 2]

# Aranacak aralıklar (log ölçeğinde olanlar katsayı olarak değişir)
SPACE = {
    'learning_rate': ('log', 0.01, 0.08),
    'num_leaves': ('int_log', 15, 255),
    'min_child_samples': ('int_log', 100, 5000),
    'feature_fraction': ('uniform', 0.2, 0.8),
    'bagging_fraction': ('uniform', 0.5, 1.0),
    'lambda_l2': ('log', 0.1, 100.0),
    'scale_pos_weight': ('log', 1.0, 20.0),
    'max_depth': ('choice', [-1, 6, 8, 12]),
    'min_gain_to_split': ('choice', [0.0, 0.1, 1.0]),
    'es_metric': ('choice', ['pr', 'auc']),
}


def sample(rng):
    p = {}
    for k, (kind, *args) in SPACE.items():
        if kind == 'log':
            p[k] = float(np.exp(rng.uniform(np.log(args[0]), np.log(args[1]))))
        elif kind == 'int_log':
            p[k] = int(round(np.exp(rng.uniform(np.log(args[0]), np.log(args[1])))))
        elif kind == 'uniform':
            p[k] = float(rng.uniform(args[0], args[1]))
        else:
            p[k] = args[0][rng.integers(len(args[0]))]
    return p


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    trials = json.load(open(PATH, encoding='utf-8')) if os.path.exists(PATH) else []
    dev = lab.load_dev()
    fold, inner = lab.fold_assignments(dev)
    y = dev['SepsisLabel'].to_numpy()
    F, _ = lab.features(dev, 'v4', lambda d: build_v4(d, group_col='Patient_ID'))

    # Referans: mevcut ayarlar aynı 3 katta (yoksa önce o ölçülür)
    if not any(t['name'] == 'mevcut' for t in trials):
        _, folds = lab.lgbm_cv(F, y, fold, inner, params={}, folds_to_run=SEARCH_FOLDS, quiet=True)
        trials.append({'name': 'mevcut', 'params': {}, 'roc_auc': float(np.mean([f['roc_auc'] for f in folds])),
                       'pr_auc': float(np.mean([f['pr_auc'] for f in folds])), 'folds': folds})
    ref = next(t for t in trials if t['name'] == 'mevcut')['roc_auc']
    lab.log(f'Referans (mevcut ayarlar, kat 1–3): ROC-AUC {ref:.4f}')

    rng = np.random.default_rng(1000 + len(trials))
    for i in range(n_trials):
        params = sample(rng)
        t0 = time.time()
        _, folds = lab.lgbm_cv(F, y, fold, inner, params=dict(params), folds_to_run=SEARCH_FOLDS, quiet=True)
        auc = float(np.mean([f['roc_auc'] for f in folds]))
        trials.append({'name': f'deneme{len(trials)}', 'params': params, 'roc_auc': auc,
                       'pr_auc': float(np.mean([f['pr_auc'] for f in folds])), 'folds': folds})
        json.dump(trials, open(PATH, 'w', encoding='utf-8'), indent=2)
        best = max(t['roc_auc'] for t in trials)
        lab.log(f'[{i + 1}/{n_trials}] ROC-AUC {auc:.4f} ({auc - ref:+.4f}) · en iyi {best:.4f} · '
                f'{(time.time() - t0) / 60:.1f} dk · ' + ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}'
                                                                 for k, v in params.items()))

    lab.log('\nEn iyi 5 deneme:')
    for t in sorted(trials, key=lambda t: -t['roc_auc'])[:5]:
        lab.log(f"  {t['name']:10s} ROC-AUC {t['roc_auc']:.4f} ({t['roc_auc'] - ref:+.4f})  PR-AUC {t['pr_auc']:.4f}  "
                f"{json.dumps(t['params'])}")


if __name__ == '__main__':
    main()
