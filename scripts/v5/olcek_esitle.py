"""
v5 skorlarını v4 ölçeğine eşler (bkz. olcek.py) ve ölçeğe bağlı her şeyi yeni ölçekte günceller.

Yalnızca VALİDASYON hastaları kullanılır (test hastalarına dokunulmaz):
  1. v4 ve v5 modellerinin validasyon tahminleri → yüzdelik eşleme → models/v5/score_map.json
  2. Kalibratör eşlenmiş skorlar üzerinde yeniden fit edilir → calibrator.pkl
  3. Eşikler (F1, tarama) ve klinik tutarlılık testi eşlenmiş ölçeğe çevrilir → model_card.json, reliability_stats.json

  py -3.10 scripts/v5/olcek_esitle.py      (train_hybrid.py v5 eğitiminden sonra bir kez)
"""

import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from features_v4 import build_v4  # noqa: E402
from gbm_numpy import NumpyGBM  # noqa: E402
from olcek import ScoreMap  # noqa: E402
from train_v3 import split_patients  # noqa: E402

V4, V5 = (os.path.join(ROOT, 'models', v) for v in ('v4', 'v5'))
QUANTILES = np.linspace(0, 1, 2001)


def load_model(directory):
    from tensorflow import keras
    card = json.load(open(os.path.join(directory, 'model_card.json'), encoding='utf-8'))
    return {'gbm': NumpyGBM.load(os.path.join(directory, 'trees.json')),
            'scaler': pickle.load(open(os.path.join(directory, 'scaler.pkl'), 'rb')),
            'mlp': keras.models.load_model(os.path.join(directory, 'mlp.keras')),
            'w': float(card['ensemble_weight_lightgbm'])}


def predict(m, F):
    return m['w'] * m['gbm'].predict(F.to_numpy()) + (1 - m['w']) * m['mlp'].predict(
        m['scaler'].transform(F), batch_size=16384, verbose=0).ravel()


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    val = df[df['Patient_ID'].isin(split_patients(df)['val'])].reset_index(drop=True)
    y = val['SepsisLabel'].to_numpy()
    F = build_v4(val, group_col='Patient_ID')
    m4, m5 = load_model(V4), load_model(V5)
    p4, p5 = predict(m4, F), predict(m5, F)
    print(f'Validasyon: {val["Patient_ID"].nunique():,} hasta · ROC-AUC v4 {roc_auc_score(y, p4):.4f} · v5 {roc_auc_score(y, p5):.4f}')

    # 1) Yüzdelik eşleme (x artan olmalı; tekrar eden değerler atılır)
    x, yq = np.quantile(p5, QUANTILES), np.quantile(p4, QUANTILES)
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    x, yq = x[keep], np.maximum.accumulate(yq[keep])
    x[0], yq[0], x[-1] = 0.0, 0.0, 1.0           # uçlar: 0 → 0, 1 → v4'ün en yüksek skoru
    json.dump({'x': x.tolist(), 'y': yq.tolist(), 'source': 'validasyon yüzdelik eşleme v5 → v4',
               'n_val_hours': int(len(y))}, open(os.path.join(V5, 'score_map.json'), 'w', encoding='utf-8'))
    smap = ScoreMap(x, yq)
    p5m = smap(p5)
    assert abs(roc_auc_score(y, p5m) - roc_auc_score(y, p5)) < 1e-3, 'Eşleme sırayı bozmamalı'

    # 2) Kalibratör eşlenmiş ölçekte
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p5m, y)
    pickle.dump(cal, open(os.path.join(V5, 'calibrator.pkl'), 'wb'))

    # 3) Eşikler ve klinik tutarlılık testi
    card_path, stats_path = os.path.join(V5, 'model_card.json'), os.path.join(V5, 'reliability_stats.json')
    card, stats = json.load(open(card_path, encoding='utf-8')), json.load(open(stats_path, encoding='utf-8'))
    raw_f1, raw_s80 = stats['val_best_f1_threshold'], stats['val_sens80_threshold']
    t_f1, t_s80 = round(smap(raw_f1), 3), smap(raw_s80)
    stats['raw_thresholds'] = {'f1': raw_f1, 'sens80': raw_s80}
    stats['val_best_f1_threshold'], stats['val_sens80_threshold'] = t_f1, t_s80
    for key, t in (('at_config_threshold', t_f1), ('at_val_best_f1_threshold', t_f1), ('at_val_sens80_threshold', t_s80)):
        stats['test'][key]['threshold'] = t
    for pl, t in zip(stats['patient_level'], (t_f1, t_s80)):
        pl['threshold'] = t
    stats['score_map'] = 'models/v5/score_map.json (v4 ölçeğine yüzdelik eşleme; sırayı korur)'
    card['threshold'] = t_f1

    def rows_predict(rows):
        f = build_v4(pd.DataFrame(rows)).iloc[[-1]]
        return smap(float(predict(m5, f)[0]))

    normal = {'HR': 78, 'O2Sat': 98, 'Temp': 36.8, 'Resp': 16, 'SBP': 125, 'DBP': 75, 'MAP': 92,
              'Lactate': 1.2, 'WBC': 8, 'Age': 65, 'Gender': 1}
    septic = dict(normal, HR=130, O2Sat=88, Temp=38.9, Resp=28, SBP=82, DBP=48, MAP=59, Lactate=4.5, WBC=17)
    scen = lambda v, h: [dict(v if hh > h - 6 else normal, Hour=hh) for hh in range(h + 1)]
    sanity = {'normal_vs_septic': {h: {'normal': rows_predict(scen(normal, h)), 'septic': rows_predict(scen(septic, h))}
                                   for h in (6, 24, 72)}, 'single_feature': {}}
    base = rows_predict(scen(normal, 24))
    for feat, bad in (('HR', 135), ('Temp', 39.2), ('Resp', 30), ('MAP', 58), ('SBP', 80),
                      ('O2Sat', 87), ('Lactate', 5.0), ('WBC', 20)):
        sanity['single_feature'][feat] = {'value': bad, 'delta': rows_predict(scen(dict(normal, **{feat: bad}), 24)) - base}
    sanity['passed'] = (all(v['septic'] > v['normal'] for v in sanity['normal_vs_septic'].values())
                        and sum(v['delta'] >= 0 for v in sanity['single_feature'].values()) >= 6)
    stats['clinical_sanity'] = sanity

    json.dump(stats, open(stats_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(card, open(card_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'Eşikler (ham → v4 ölçeği): F1 {raw_f1:.3f} → {t_f1:.3f} · tarama {raw_s80:.4f} → {t_s80:.4f}')
    print('Klinik tutarlılık:', json.dumps({h: {k: round(v, 3) for k, v in d.items()} for h, d in sanity['normal_vs_septic'].items()}),
          '| tek parametre:', {k: round(v['delta'] * 100, 1) for k, v in sanity['single_feature'].items()})


if __name__ == '__main__':
    main()
