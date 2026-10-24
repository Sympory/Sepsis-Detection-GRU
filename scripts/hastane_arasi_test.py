"""
Hastaneler arası (dış) doğrulama: model bir hastanede eğitilip hiç görmediği diğer hastanede test edilir.

PhysioNet 2019 verisi iki hastaneden gelir: Patient_ID < 100000 → hastane A (training_setA, 20.336 hasta),
>= 100000 → hastane B (training_setB, 20.000 hasta). Her hedef hastane T için hastalarının %30'u sabit bir
test grubu olarak ayrılır ve üç model AYNI test grubunda karşılaştırılır:

  dış      : yalnızca diğer hastanenin verisiyle eğitildi   → "başka hastaneye götürürsek ne olur?"
  iç       : T'nin kalan %70'iyle eğitildi                   → "o hastanenin kendi verisiyle ne olur?"
  birleşik : diğer hastane + T'nin %70'i                     → "yerel veri ekleyince ne olur?"

Eşikler (tarama: duyarlılık >= %80, Utility) her modelin kendi eğitim hastanesindeki iç validasyonda seçilir;
hedef hastanede eşiğin taşınıp taşınmadığı da böylece ölçülür. Ayarlar ana modelle aynıdır
(LightGBM tek seed + MLP, 0.7/0.3).

  py -3.10 scripts/hastane_arasi_test.py      → models/<sürüm>/hastane_arasi_test.json
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

from evaluate_clinical import UtilityTable
from features_v4 import V4Scaler, build_v4
from train_hybrid import LGBM_PARAMS, MONOTONE, VERSION, monotone_vector

W_GBM = 0.9 if VERSION == 'v5' else 0.7   # v5: çapraz doğrulamada kat dışında seçilen LightGBM payı
TEST_FRAC = 0.3
OUT = os.path.join(ROOT, 'models', VERSION, 'hastane_arasi_test.json')
HOSPITALS = ('A', 'B')


def log(msg):
    print(msg, flush=True)


def patient_split(df, seed=42):
    """Hasta bazlı, sepsis durumuna göre tabakalı: hedef test grubu ve iç validasyon işaretleri."""
    status = df.groupby('Patient_ID')['SepsisLabel'].max()
    rng = np.random.default_rng(seed)
    test, inner_val = {}, {}
    for p in status.index:
        test[p] = False
        inner_val[p] = rng.random() < 0.2
    for label in (0, 1):
        ids = status[status == label].index.to_numpy()
        rng.shuffle(ids)
        for p in ids[:int(round(len(ids) * TEST_FRAC))]:
            test[p] = True
    return test, inner_val


def screening_threshold(y, p, target=0.80):
    """Duyarlılığı en az `target` olan en yüksek eşik."""
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def best_utility_threshold(table, p):
    grid = np.quantile(p, np.linspace(0.50, 0.995, 80))
    scores = [table.score(p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    df['hospital'] = np.where(df['Patient_ID'] < 100000, 'A', 'B')
    test, inner = patient_split(df)
    is_test = df['Patient_ID'].map(test).to_numpy()
    iv = df['Patient_ID'].map(inner).to_numpy()
    hosp = df['hospital'].to_numpy()
    y = df['SepsisLabel'].to_numpy()
    pid = df['Patient_ID'].to_numpy()
    F = build_v4(df, group_col='Patient_ID')
    log(f'{F.shape[1]} özellik · {(time.time() - t0) / 60:.1f} dk')

    # Deney listesi: (hedef hastane, senaryo, eğitim satırları)
    experiments = []
    for T in HOSPITALS:
        S = 'B' if T == 'A' else 'A'
        experiments += [
            (T, 'dış', hosp == S),
            (T, 'iç', (hosp == T) & ~is_test),
            (T, 'birleşik', (hosp == S) | ((hosp == T) & ~is_test)),
        ]

    # 1) LightGBM — TensorFlow yüklenmeden
    pr = lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True)
    preds = {}
    for T, name, pool in experiments:
        tr, va = pool & ~iv, pool & iv
        dtrain = lgb.Dataset(F[tr], y[tr], free_raw_data=False)
        dval = lgb.Dataset(F[va], y[va], reference=dtrain, free_raw_data=False)
        params = dict(LGBM_PARAMS, seed=42, bagging_seed=42, feature_fraction_seed=42)
        if MONOTONE:
            params['monotone_constraints'] = monotone_vector(list(F.columns))
        b = lgb.train(params, dtrain,
                      num_boost_round=4000, valid_sets=[dval], feval=pr,
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        rows = va | ((hosp == T) & is_test)          # yalnızca gereken satırlar tahmin edilir
        p = np.full(len(df), np.nan)
        p[rows] = b.predict(F[rows], num_iteration=b.best_iteration)
        preds[(T, name)] = {'gbm': p}
        log(f'  LightGBM  hedef {T} · {name:8s}: {b.best_iteration} ağaç · {(time.time() - t0) / 60:.1f} dk')

    # 2) Sinir ağı
    import tensorflow as tf
    from tensorflow import keras
    from train_hybrid import build_mlp
    for T, name, pool in experiments:
        tf.keras.utils.set_random_seed(42)
        tr, va = pool & ~iv, pool & iv
        rows = va | ((hosp == T) & is_test)
        scaler = V4Scaler().fit(F[tr])
        Xtr, Xva = scaler.transform(F[tr]), scaler.transform(F[va])
        mlp = build_mlp(Xtr.shape[1])
        mlp.fit(Xtr, y[tr], validation_data=(Xva, y[va]), epochs=30, batch_size=4096,
                class_weight={0: 1.0, 1: 5.0}, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=4, restore_best_weights=True),
                           keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)])
        p = np.full(len(df), np.nan)
        p[rows] = mlp.predict(scaler.transform(F[rows]), batch_size=16384, verbose=0).ravel()
        preds[(T, name)]['mlp'] = p
        del Xtr, Xva
        log(f'  Sinir ağı hedef {T} · {name:8s} · {(time.time() - t0) / 60:.1f} dk')

    # 3) Değerlendirme — her hedef için aynı test hastaları
    res = {'hospitals': {}, 'results': {}, 'test_fraction': TEST_FRAC}
    for H in HOSPITALS:
        m = hosp == H
        res['hospitals'][H] = {'patients': int(pd.unique(pid[m]).size),
                               'sepsis_patient_rate': float(df[m].groupby('Patient_ID')['SepsisLabel'].max().mean())}
    for T, name, pool in experiments:
        te = (hosp == T) & is_test
        va = pool & iv
        p_all = W_GBM * preds[(T, name)]['gbm'] + (1 - W_GBM) * preds[(T, name)]['mlp']
        p_te, p_va = p_all[te], p_all[va]
        thr_screen = screening_threshold(y[va], p_va)
        thr_util = best_utility_threshold(UtilityTable(y[va], pid[va]), p_va)
        util_te = UtilityTable(y[te], pid[te])
        a_s = p_te >= thr_screen
        r = {
            'train_patients': int(pd.unique(pid[pool]).size),
            'test_patients': int(pd.unique(pid[te]).size),
            'roc_auc': float(roc_auc_score(y[te], p_te)),
            'pr_auc': float(average_precision_score(y[te], p_te)),
            'lightgbm_roc_auc': float(roc_auc_score(y[te], preds[(T, name)]['gbm'][te])),
            'mlp_roc_auc': float(roc_auc_score(y[te], preds[(T, name)]['mlp'][te])),
            'prevalence': float(y[te].mean()),
            'screen_threshold': thr_screen,
            'screen_sensitivity': float(a_s[y[te] == 1].mean()),
            'screen_specificity': float(1 - a_s[y[te] == 0].mean()),
            'utility_threshold': thr_util,
            'utility': util_te.score(p_te >= thr_util),
        }
        res['results'].setdefault(T, {})[name] = r
        log(f"hedef {T} · {name:8s}: ROC-AUC {r['roc_auc']:.3f} · PR-AUC {r['pr_auc']:.3f} · "
            f"Utility {r['utility']:.3f} · tarama duyarlılık {r['screen_sensitivity']:.0%} / özgüllük {r['screen_specificity']:.0%}")
    res['minutes'] = (time.time() - t0) / 60
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log(f'Kaydedildi: {OUT} · {res["minutes"]:.1f} dk')


if __name__ == '__main__':
    main()
