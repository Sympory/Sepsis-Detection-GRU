"""
Hasta bazlı test: modelin hiç görmediği test hastalarını saat saat uygulama mantığıyla oynatır
(model + klinik kurallar + Yetersiz Veri kapısı) ve sepsis hastalarının kaçının klinik
tanıdan önce alarm aldığını, sepsis olmayanların kaçına yanlış alarm verildiğini ölçer.

Kullanım:  py -3.10 scripts/hasta_bazli_test.py
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np, pandas as pd
from features import build_hourly_features, make_windows
from train_v3 import split_patients
from config import CLINICAL_OVERRIDES, OPTIMAL_THRESHOLD
from reliability import CORE_VITALS, MIN_CORE_VITALS
from tensorflow import keras

df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
test_ids = split_patients(df)['test']
d = df[df.Patient_ID.isin(test_ids)].reset_index(drop=True)
from config import FEATURE_SET, MODEL_INFO
groups = d.groupby('Patient_ID', sort=False).indices
if FEATURE_SET == 'v4':
    # Uygulamadaki hibrit model: zaman serisi özellikleri → LightGBM (NumPy) + sinir ağı
    from features_v4 import build_v4
    from gbm_numpy import NumpyGBM
    F = build_v4(d, group_col='Patient_ID')
    gbm = NumpyGBM.load(os.path.join(ROOT, 'models', 'v4', 'trees.json'))
    scaler = pickle.load(open(os.path.join(ROOT, 'models', 'v4', 'scaler.pkl'), 'rb'))
    mlp = keras.models.load_model(os.path.join(ROOT, 'models', 'v4', 'mlp.keras'))
    w = float(MODEL_INFO['ensemble_weight_lightgbm'])
    p = w * gbm.predict(F.to_numpy()) + (1 - w) * mlp.predict(scaler.transform(F), batch_size=16384, verbose=0).ravel()
else:
    tr = pickle.load(open(os.path.join(ROOT, 'models', 'v3', 'transformer.pkl'), 'rb'))
    model = keras.models.load_model(os.path.join(ROOT, 'models', 'v3', 'model.keras'))
    X = tr.transform(build_hourly_features(d, group_col='Patient_ID'))
    Xw, idx_all = [], []
    for pid, idx in groups.items():
        idx = np.sort(idx); Xw.append(make_windows(X[idx], 6)); idx_all.append(idx)
    Xw = np.concatenate(Xw); idx_all = np.concatenate(idx_all)
    p = np.empty(len(d)); p[idx_all] = model.predict(Xw, batch_size=8192, verbose=0).ravel()
d['p'] = p

# Uygulamadaki klinik kural puanı (o saatte girilen ham değerlerle)
pts = np.zeros(len(d))
for name, rule in CLINICAL_OVERRIDES.items():
    f = name.rsplit('_', 1)[0]; v = d[f]
    hit = (v < rule['threshold']) if rule['direction'] == 'below' else (v > rule['threshold'])
    pts += hit.fillna(False).to_numpy() * rule['red_flags']
d['pts'] = pts
adj = p.copy()
adj = np.where(pts >= 5, np.maximum(adj, 0.60), np.where(pts >= 3, np.maximum(adj, 0.35), np.where(pts >= 2, np.maximum(adj, 0.25), adj)))
d['adj'] = adj

# Veri kapısı: o saatte girilen temel vital sayısı (MAP yoksa SBP+DBP'den)
mapok = d['MAP'].notna() | (d['SBP'].notna() & d['DBP'].notna())
core = sum(d[c].notna().astype(int) for c in ['HR', 'O2Sat', 'Temp', 'Resp']) + mapok.astype(int)
d['gate_now'] = core >= MIN_CORE_VITALS
# Alternatif: son 4 saatte ölçülmüş temel vitaller (daha gerçekçi kapı)
recent = sum(d.groupby('Patient_ID')[c].transform(lambda s: s.notna().rolling(4, min_periods=1).max()).astype(int) for c in ['HR', 'O2Sat', 'Temp', 'Resp'])
recent = recent + mapok.groupby(d.Patient_ID).transform(lambda s: s.rolling(4, min_periods=1).max()).astype(int)
d['gate_recent'] = recent >= MIN_CORE_VITALS

rules = {
    f'Model yalnız (eşik {OPTIMAL_THRESHOLD})': d.p >= OPTIMAL_THRESHOLD,
    'Model yalnız (tarama eşiği)': d.p >= float(json.load(open(os.path.join(ROOT, 'models', FEATURE_SET, 'reliability_stats.json'), encoding='utf-8'))['val_sens80_threshold']),
    'Uygulama: Yüksek+ (model+klinik kural, veri kapısı)': (d.adj >= 0.35) & d.gate_now,
    'Uygulama: Orta+ (model+klinik kural, veri kapısı)': (d.adj >= 0.20) & d.gate_now,
}
out = {'n_test_patients': int(d.Patient_ID.nunique()),
       'gate_now_hours_ok': float(d.gate_now.mean()), 'gate_recent_hours_ok': float(d.gate_recent.mean())}
g = d.groupby('Patient_ID')
status = g.SepsisLabel.max()
first_pos = g.apply(lambda x: int(np.argmax(x.SepsisLabel.values == 1)) if x.SepsisLabel.max() == 1 else -1)
sep_ids = status[status == 1].index
late_ids = [pid for pid in sep_ids if first_pos[pid] >= 6]  # yatıştan en az 6 saat sonra etiketlenen (erken uyarı anlamlı)
out['sepsis_patients'] = len(sep_ids); out['sepsis_patients_label_after_6h'] = len(late_ids)
out['non_sepsis_patients'] = int((status == 0).sum())
for name, alert in rules.items():
    d['_a'] = alert.to_numpy()
    caught_label, caught_clinical, leads = 0, 0, []
    for pid in late_ids:
        x = d.loc[groups[pid]]
        fp_ = first_pos[pid]; a = x._a.values
        clinical_onset = fp_ + 6  # etiket, klinik başlangıçtan 6 saat önce başlar
        if a[:fp_ + 1].any(): caught_label += 1
        hit = np.where(a[:min(clinical_onset + 1, len(a))])[0]
        if len(hit): caught_clinical += 1; leads.append(clinical_onset - hit[0])
    neg = d[d.Patient_ID.isin(status[status == 0].index)]
    neg_alerted = neg.groupby('Patient_ID')._a.any().mean()
    alerts_per_day = neg._a.sum() / (len(neg) / 24)
    out[name] = {
        'sepsis_caught_by_label_start': caught_label / len(late_ids),
        'sepsis_caught_by_clinical_onset': caught_clinical / len(late_ids),
        'median_lead_hours': float(np.median(leads)) if leads else None,
        'non_sepsis_ever_alerted': float(neg_alerted),
        'alert_hours_per_non_sepsis_patient_day': float(alerts_per_day),
    }
# Örnek hastalar: geç etiketlenen 5 sepsis hastası saat saat
rng = np.random.default_rng(7)
examples = []
for pid in rng.choice(late_ids, 5, replace=False):
    x = d.loc[groups[pid]]; fp_ = first_pos[pid]
    lo, hi = max(0, fp_ - 10), min(len(x), fp_ + 8)
    examples.append({'pid': int(pid), 'label_start_hour': int(fp_), 'clinical_onset_hour': int(fp_) + 6,
                     'hours': [(int(h), round(float(pp) * 100, 1), int(pt), bool(gt)) for h, pp, pt, gt in
                               zip(range(lo, hi), x.p.values[lo:hi], x.pts.values[lo:hi], x.gate_now.values[lo:hi])]})
out['examples'] = examples
print(json.dumps(out, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
