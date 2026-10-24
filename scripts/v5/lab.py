"""
v5 deney altyapısı — bütün model geliştirme deneylerinin ortak zemini.

Kurallar:
  1. Kilitli test seti: train_v3.split_patients'in %20'lik test hastaları. Deneylerde HİÇ kullanılmaz;
     yalnızca en sonda bir kez (final) açılır.
  2. Geliştirme seti: kalan %80 hasta, sepsis durumuna göre tabakalı 5 kata bölünür (seed 42, her deneyde aynı).
  3. Her deney aynı katlarda koşar; böylece iki deney kat kat (eşleştirilmiş) karşılaştırılabilir.
  4. Özellikler bir kez hesaplanıp data/processed/ altına önbelleğe yazılır (her deneyde yeniden hesaplanmaz).

Kullanım: scripts/v5/deney.py bu modülü çağırır.
"""

import lightgbm as lgb  # noqa: F401  (Windows'ta diğer kütüphanelerden önce yüklenmeli)

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from train_v3 import split_patients

K = 5
SEED = 42
CACHE_DIR = os.path.join(ROOT, 'data', 'processed')
OUT_DIR = os.path.join(ROOT, 'models', 'v5')
LOG_PATH = os.path.join(OUT_DIR, 'deneyler.json')

# Ana modelle aynı LightGBM ayarları (scripts/train_hybrid.py) — temel çizgi
BASE_LGBM = dict(objective='binary', metric='None', learning_rate=0.03, num_leaves=31, min_child_samples=1000,
                 feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0,
                 scale_pos_weight=5.0, verbose=-1)


def log(msg):
    print(msg, flush=True)


# ============================================================================
# VERİ VE BÖLME
# ============================================================================

def load_dev():
    """Geliştirme setini (test hastaları hariç) saat sırasıyla döndürür."""
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    test_ids = split_patients(df, seed=SEED)['test']
    dev = df[~df['Patient_ID'].isin(test_ids)].reset_index(drop=True)
    return dev


def fold_assignments(dev):
    """Her hastaya 0..K-1 arası kat ve %20 iç validasyon işareti (erken durdurma için)."""
    status = dev.groupby('Patient_ID')['SepsisLabel'].max()
    rng = np.random.default_rng(SEED)
    fold, inner = {}, {}
    for label in (0, 1):
        ids = status[status == label].index.to_numpy()
        rng.shuffle(ids)
        for i, p in enumerate(ids):
            fold[p] = i % K
            inner[p] = rng.random() < 0.2
    return dev['Patient_ID'].map(fold).to_numpy(), dev['Patient_ID'].map(inner).to_numpy()


def features(dev, name, builder):
    """Özellik matrisini önbellekten okur; yoksa `builder(dev)` ile hesaplayıp kaydeder."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'v5_ozellik_{name}.npy')
    cols_path = path.replace('.npy', '_sutunlar.json')
    if os.path.exists(path) and os.path.exists(cols_path):
        F = np.load(path, mmap_mode='r')
        with open(cols_path, encoding='utf-8') as f:
            cols = json.load(f)
        if F.shape[0] == len(dev):
            log(f'Özellikler önbellekten: {name} ({F.shape[1]} sütun)')
            return np.asarray(F), cols
    t0 = time.time()
    frame = builder(dev)
    F = frame.to_numpy(np.float32)
    np.save(path, F)
    with open(cols_path, 'w', encoding='utf-8') as f:
        json.dump(list(frame.columns), f)
    log(f'Özellikler hesaplandı: {name} ({F.shape[1]} sütun, {(time.time() - t0) / 60:.1f} dk) → önbelleğe yazıldı')
    return F, list(frame.columns)


# ============================================================================
# ÇAPRAZ DOĞRULAMA
# ============================================================================

def lgbm_cv(F, y, fold, inner, params=None, seeds=(SEED,), weight=None, monotone=None, folds_to_run=None, quiet=False):
    """
    5 katlı LightGBM. Dönüş: (her satır için kat-dışı tahmin, kat sonuçları).
    Erken durdurma iç validasyon PR-AUC'sine göre; test katı karar için hiç kullanılmaz.
    """
    params = dict(BASE_LGBM, **(params or {}))
    # Erken durdurma ölçüsü: 'pr' (PR-AUC, varsayılan) veya 'auc' (ROC-AUC, daha az dalgalı)
    es_metric = params.pop('es_metric', 'pr')
    patience = params.pop('patience', 200)
    if monotone is not None:
        params['monotone_constraints'] = monotone
    if es_metric == 'auc':
        feval = lambda p, d: ('roc_auc', roc_auc_score(d.get_label(), p), True)
    else:
        feval = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)
    oof = np.zeros(len(y))
    folds = []
    for k in (folds_to_run if folds_to_run is not None else range(K)):
        te, tr, va = fold == k, (fold != k) & ~inner, (fold != k) & inner
        p = np.zeros(te.sum())
        trees = []
        for s in seeds:
            dtrain = lgb.Dataset(F[tr], y[tr], weight=None if weight is None else weight[tr], free_raw_data=False)
            dval = lgb.Dataset(F[va], y[va], reference=dtrain, free_raw_data=False)
            b = lgb.train(dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s), dtrain,
                          num_boost_round=6000, valid_sets=[dval], feval=feval,
                          callbacks=[lgb.early_stopping(patience, verbose=False)])
            p += b.predict(F[te], num_iteration=b.best_iteration) / len(seeds)
            trees.append(b.best_iteration)
        oof[te] = p
        folds.append({'roc_auc': float(roc_auc_score(y[te], p)), 'pr_auc': float(average_precision_score(y[te], p)),
                      'trees': trees})
        if not quiet:
            log(f'  kat {k + 1}: ROC-AUC {folds[-1]["roc_auc"]:.4f}  PR-AUC {folds[-1]["pr_auc"]:.4f}  ağaç {trees}')
    return oof, folds


def summarize(folds):
    out = {}
    for m in ('roc_auc', 'pr_auc'):
        v = np.array([f[m] for f in folds])
        out[m] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1))}
    return out


# ============================================================================
# KAYIT VE KARŞILAŞTIRMA
# ============================================================================

def load_log():
    if not os.path.exists(LOG_PATH):
        return {}
    with open(LOG_PATH, encoding='utf-8') as f:
        return json.load(f)


def save_result(name, description, folds, extra=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_log()
    results[name] = {'description': description, 'folds': folds, 'summary': summarize(folds),
                     'date': time.strftime('%Y-%m-%d %H:%M'), **(extra or {})}
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def compare(name, reference='baseline'):
    """
    Eşleştirilmiş karşılaştırma: aynı katta yeni deney − referans.
    Karar kuralı: 5 katın en az 4'ünde iyileşme VE ortalama fark ≥ 0.002 ROC-AUC → değişiklik kalır.
    """
    results = load_log()
    if reference not in results or name == reference:
        return None
    a, b = results[name]['folds'], results[reference]['folds']
    lines, verdict = [], {}
    for m in ('roc_auc', 'pr_auc'):
        d = np.array([x[m] - y[m] for x, y in zip(a, b)])
        better = int((d > 0).sum())
        lines.append(f'  {m.upper():8s} fark {d.mean():+.4f} (kat kat: {" ".join(f"{v:+.4f}" for v in d)}) · '
                     f'{better}/{K} katta daha iyi')
        verdict[m] = {'mean_diff': float(d.mean()), 'folds_better': better}
    keep = verdict['roc_auc']['folds_better'] >= 4 and verdict['roc_auc']['mean_diff'] >= 0.002
    lines.append(f'  KARAR: {"KALIR ✓" if keep else "kazanç anlamlı değil ✗"} (kural: ≥4/5 kat ve ≥ +0.002 ROC-AUC)')
    return '\n'.join(lines), keep
