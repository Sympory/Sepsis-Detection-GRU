"""
Klinik skorlarla karşılaştırma ve PhysioNet 2019 Utility skoru.

Aynı test hastalarında:
  - Hibrit v4 ve GRU v3
  - Hastanelerde kullanılan skorlar: SIRS, qSOFA (kısmi), NEWS2 (kısmi)
için ROC-AUC, PR-AUC, standart kesim noktalarında duyarlılık/özgüllük ve
PhysioNet 2019 Challenge'ın resmi Normalized Utility skoru hesaplanır.

Modellerin Utility eşiği VALİDASYON hastalarında seçilir.
Not: GKS (bilinç) verisi olmadığı için qSOFA ve NEWS2 kısmi hesaplanır.

  py -3.10 scripts/evaluate_clinical.py     → models/v4/clinical_comparison.json
"""

import json
import os
import pickle
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_hourly_features, make_windows
from features_v4 import build_v4
from gbm_numpy import NumpyGBM
from train_v3 import split_patients

OUT = os.path.join(ROOT, 'models', 'v4', 'clinical_comparison.json')


# ============================================================================
# PhysioNet 2019 Utility (resmi tanım: Reyna ve ark., 2020)
# ============================================================================

def patient_utility(labels, preds, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                    max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """Tek hasta için toplam fayda. Etiket, klinik başlangıçtan 6 saat önce 1 olur."""
    if labels.any():
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
    m1 = max_u_tp / (dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal)
    b3 = -m3 * dt_optimal
    u = np.zeros(len(labels))
    for t in range(len(labels)):
        if t <= t_sepsis + dt_late:
            if is_septic and preds[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m1 * (t - t_sepsis) + b1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m2 * (t - t_sepsis) + b2
            elif is_septic and not preds[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m3 * (t - t_sepsis) + b3
            elif not is_septic and preds[t]:
                u[t] = u_fp
            else:
                u[t] = u_tn
    return u.sum()


class UtilityTable:
    """
    Her satır için 'alarm verilirse' ve 'verilmezse' faydası bir kez hesaplanır;
    böylece farklı eşikler vektörel olarak değerlendirilir. Sonuç resmi tanımla aynıdır.
    """

    def __init__(self, y, pid):
        self.u_alarm = np.zeros(len(y))
        self.u_quiet = np.zeros(len(y))
        self.best = self.inaction = 0.0
        for g, idx in pd.Series(np.arange(len(y))).groupby(pid, sort=False).indices.items():
            lab = y[idx].astype(bool)
            n = len(lab)
            ones, zeros = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
            self.u_alarm[idx] = [patient_utility_row(lab, t, True) for t in range(n)]
            self.u_quiet[idx] = [patient_utility_row(lab, t, False) for t in range(n)]
            best_pred = zeros.copy()
            if lab.any():
                t_sepsis = int(np.argmax(lab)) + 6
                best_pred[max(0, t_sepsis - 12): min(t_sepsis + 3 + 1, n)] = True
            self.best += np.where(best_pred, self.u_alarm[idx], self.u_quiet[idx]).sum()
            self.inaction += self.u_quiet[idx].sum()

    def score(self, alarms):
        observed = np.where(alarms, self.u_alarm, self.u_quiet).sum()
        return float((observed - self.inaction) / (self.best - self.inaction))


def patient_utility_row(labels, t, alarm):
    """t. saatin faydası (patient_utility'nin tek satırlık hali)."""
    return _row_utility(labels, t, alarm)


def _row_utility(labels, t, alarm, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    if labels.any():
        is_septic, t_sepsis = True, int(np.argmax(labels)) - dt_optimal
    else:
        is_septic, t_sepsis = False, float('inf')
    if t > t_sepsis + dt_late:
        return 0.0
    m1 = max_u_tp / (dt_optimal - dt_early); b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal); b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal); b3 = -m3 * dt_optimal
    if is_septic and alarm:
        return max(m1 * (t - t_sepsis) + b1, u_fp) if t <= t_sepsis + dt_optimal else m2 * (t - t_sepsis) + b2
    if is_septic:
        return 0.0 if t <= t_sepsis + dt_optimal else m3 * (t - t_sepsis) + b3
    return u_fp if alarm else u_tn


# ============================================================================
# Klinik skorlar (ileri doldurulmuş değerlerle)
# ============================================================================

def clinical_scores(df):
    F = df.groupby('Patient_ID', sort=False)[['HR', 'Temp', 'Resp', 'SBP', 'O2Sat', 'WBC', 'PaCO2', 'FiO2']].ffill()
    sirs = ((F.HR > 90).astype(int) + ((F.Temp > 38) | (F.Temp < 36)).astype(int)
            + ((F.Resp > 20) | (F.PaCO2 < 32)).astype(int) + ((F.WBC > 12) | (F.WBC < 4)).astype(int))
    qsofa = (F.Resp >= 22).astype(int) + (F.SBP <= 100).astype(int)

    def band(x, edges, scores):
        out = pd.Series(0, index=x.index)
        for (lo, hi), sc in zip(edges, scores):
            out[(x >= lo) & (x <= hi)] = sc
        return out
    news = (band(F.Resp, [(-1e9, 8), (9, 11), (21, 24), (25, 1e9)], [3, 1, 2, 3])
            + band(F.O2Sat, [(-1e9, 91), (92, 93), (94, 95)], [3, 2, 1])
            + (F.FiO2 > 0.21).astype(int) * 2
            + band(F.SBP, [(-1e9, 90), (91, 100), (101, 110), (220, 1e9)], [3, 2, 1, 3])
            + band(F.HR, [(-1e9, 40), (41, 50), (91, 110), (111, 130), (131, 1e9)], [3, 1, 1, 2, 3])
            + band(F.Temp, [(-1e9, 35.0), (35.05, 36.0), (38.05, 39.0), (39.05, 1e9)], [3, 1, 1, 2]))
    return {'SIRS': (sirs.to_numpy(), 2), 'qSOFA (kısmi)': (qsofa.to_numpy(), 2), 'NEWS2 (kısmi)': (news.to_numpy(), 5)}


def at_cut(y, alarm):
    tp = int((alarm & (y == 1)).sum()); fn = int((~alarm & (y == 1)).sum())
    tn = int((~alarm & (y == 0)).sum()); fp = int((alarm & (y == 0)).sum())
    return {'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1), 'ppv': tp / max(tp + fp, 1)}


def main():
    from tensorflow import keras
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    pid = df['Patient_ID'].to_numpy()
    va, te = (np.isin(pid, list(parts[k])) for k in ('val', 'test'))
    y = df['SepsisLabel'].to_numpy()

    print('Hibrit v4 tahminleri...', flush=True)
    card = json.load(open(os.path.join(ROOT, 'models', 'v4', 'model_card.json'), encoding='utf-8'))
    w = card['ensemble_weight_lightgbm']
    F = build_v4(df, group_col='Patient_ID')
    gbm = NumpyGBM.load(os.path.join(ROOT, 'models', 'v4', 'trees.json'))
    scaler = pickle.load(open(os.path.join(ROOT, 'models', 'v4', 'scaler.pkl'), 'rb'))
    mlp = keras.models.load_model(os.path.join(ROOT, 'models', 'v4', 'mlp.keras'))
    mask = va | te
    p_v4 = np.full(len(df), np.nan)
    p_v4[mask] = w * gbm.predict(F[mask].to_numpy()) + (1 - w) * mlp.predict(scaler.transform(F[mask]), batch_size=16384, verbose=0).ravel()
    del F

    print('GRU v3 tahminleri...', flush=True)
    tr3 = pickle.load(open(os.path.join(ROOT, 'models', 'v3', 'transformer.pkl'), 'rb'))
    gru = keras.models.load_model(os.path.join(ROOT, 'models', 'v3', 'model.keras'))
    X3 = tr3.transform(build_hourly_features(df, group_col='Patient_ID'))
    p_v3 = np.full(len(df), np.nan)
    idx_all, wins = [], []
    for g, idx in df.groupby('Patient_ID', sort=False).indices.items():
        if mask[idx[0]]:
            idx = np.sort(idx); wins.append(make_windows(X3[idx], 6)); idx_all.append(idx)
    p_v3[np.concatenate(idx_all)] = gru.predict(np.concatenate(wins), batch_size=8192, verbose=0).ravel()

    scores = clinical_scores(df)
    print('Utility tabloları...', flush=True)
    U_val, U_test = UtilityTable(y[va], pid[va]), UtilityTable(y[te], pid[te])
    result = {'note': 'Test hastaları (8.066). qSOFA ve NEWS2 bilinç (GKS) olmadan kısmi hesaplandı. '
                      'Utility eşikleri validasyonda seçildi.', 'models': {}}

    for name, p in (('Hibrit v4', p_v4), ('GRU v3', p_v3)):
        grid = np.quantile(p[va], np.linspace(0.5, 0.995, 60))
        utils = [U_val.score(p[va] >= t) for t in grid]
        t_best = float(grid[int(np.argmax(utils))])
        result['models'][name] = {
            'roc_auc': float(roc_auc_score(y[te], p[te])), 'pr_auc': float(average_precision_score(y[te], p[te])),
            'utility_threshold_val': t_best, 'utility_val': float(max(utils)),
            'utility_test': U_test.score(p[te] >= t_best),
            **{f'at_utility_threshold_{k}': v for k, v in at_cut(y[te], p[te] >= t_best).items()},
        }
        print(f"  {name}: AUC {result['models'][name]['roc_auc']:.4f} · Utility(test) {result['models'][name]['utility_test']:.3f}", flush=True)

    for name, (sc, cut) in scores.items():
        s_te = sc[te].astype(float)
        result['models'][name] = {
            'roc_auc': float(roc_auc_score(y[te], s_te)), 'pr_auc': float(average_precision_score(y[te], s_te)),
            'cutoff': f'≥ {cut}', 'utility_test': U_test.score(sc[te] >= cut),
            **{f'at_cutoff_{k}': v for k, v in at_cut(y[te], sc[te] >= cut).items()},
        }
        print(f"  {name}: AUC {result['models'][name]['roc_auc']:.4f} · Utility {result['models'][name]['utility_test']:.3f}", flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('Kaydedildi:', OUT)


if __name__ == '__main__':
    main()
