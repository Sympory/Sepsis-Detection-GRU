"""
Tek komutla deney: aynı 5 katlı çapraz doğrulamada LightGBM eğitir, sonucu kaydeder, temel çizgiyle karşılaştırır.

  py -3.10 scripts/v5/deney.py baseline        # temel çizgi (mevcut v4 özellikleri ve ayarları)
  py -3.10 scripts/v5/deney.py --liste         # tanımlı deneyler ve kayıtlı sonuçlar

Yeni bir deney eklemek için aşağıdaki DENEYLER sözlüğüne bir satır eklenir.
Sonuçlar: models/v5/deneyler.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402  (lightgbm'i ilk yükler)

import time  # noqa: E402

import numpy as np  # noqa: E402

from features_v4 import build_v4  # noqa: E402
from features_v5 import GROUPS  # noqa: E402

# Özellik grubu adı → üretici. 'v4' mevcut 281 özellik; diğerleri features_v5.py'deki ek gruplar.
BUILDERS = {'v4': lambda dev: build_v4(dev, group_col='Patient_ID'),
            **{name: (lambda f: (lambda dev: f(dev, group_col='Patient_ID')))(f) for name, f in GROUPS.items()}}

# ad: (açıklama, özellik grupları, LightGBM ayar değişiklikleri)
DENEYLER = {
    'baseline': ('Mevcut model: v4 özellikleri (281), ana LightGBM ayarları', ['v4'], {}),
    # Aşama 2 — her ek grup tek başına
    'v4+uzun': ('48 ve 72 saatlik pencere istatistikleri', ['v4', 'uzun'], {}),
    'v4+egim': ('6 ve 24 saatlik trend eğimi', ['v4', 'egim'], {}),
    'v4+esik': ('Son 6/24 saatte anormal saat sayısı ve kesintisiz süre', ['v4', 'esik'], {}),
    'v4+sapma': ('Hastanın kendi normaline göre z-skoru ve ilk değerden fark', ['v4', 'sapma'], {}),
    'v4+olcum': ('Test isteme yoğunluğu', ['v4', 'olcum'], {}),
    # Aşama 3 — ayar aramasının en iyi iki sonucu (scripts/v5/ayar_ara.py), tam 5 katta doğrulama
    'ayar_d17': ('Ayar arama #17: derin ağaçlar (240 yaprak), ROC-AUC ile erken durdurma', ['v4'],
                 {'learning_rate': 0.0224, 'num_leaves': 240, 'min_child_samples': 1308, 'feature_fraction': 0.652,
                  'bagging_fraction': 0.607, 'lambda_l2': 35.3, 'scale_pos_weight': 1.04, 'max_depth': 12,
                  'min_gain_to_split': 1.0, 'es_metric': 'auc'}),
    'ayar_d12': ('Ayar arama #12: yavaş öğrenme (0.011), az özellik örneklemi (%21)', ['v4'],
                 {'learning_rate': 0.0106, 'num_leaves': 32, 'min_child_samples': 2274, 'feature_fraction': 0.209,
                  'bagging_fraction': 0.669, 'lambda_l2': 4.40, 'scale_pos_weight': 1.87, 'max_depth': 12,
                  'min_gain_to_split': 0.1}),
}
D12 = DENEYLER['ayar_d12'][2]
# Aşama 5 — eğitim hedefi: satır ağırlıkları, referans ayar_d12. agirlik = (klinik başlangıç sonrası, erken uyarı penceresi)
DENEYLER.update({
    'agirlik_sonra03': ('Klinik başlangıç sonrası saatler ×0.3 (model erken işaretlere odaklansın)', ['v4'],
                        dict(D12, agirlik=(0.3, 1.0), referans='ayar_d12')),
    'agirlik_erken2': ('Erken uyarı penceresi (etiketin ilk 6 saati) ×2', ['v4'],
                       dict(D12, agirlik=(1.0, 2.0), referans='ayar_d12')),
    # Aşama 6 — klinik tutarlılık: değer kötüleştikçe risk düşemez (-1: değer düştükçe risk artar)
    'monoton': ('Monoton kısıt: MAP, SBP, SpO2 ↓ ve laktat ↑ iken risk düşemez', ['v4'],
                dict(D12, monoton={'MAP': -1, 'SBP': -1, 'O2Sat': -1, 'Lactate': 1}, referans='ayar_d12')),
})


def monotone_vector(cols, rules):
    """Değişkenin kendisi ve pencere istatistikleri (ortalama/min/maks) kısıtlanır; ölçüm sıklığı, süre, değişim serbest."""
    vec = []
    for c in cols:
        base, _, suffix = c.partition('_')
        ok = suffix == '' or any(suffix.startswith(s) for s in ('mean', 'min', 'max'))
        vec.append(rules.get(base, 0) if ok else 0)
    return vec


def row_weights(dev, after, early):
    """Etiket başlangıcından itibaren saat: 0–5 erken uyarı penceresi, ≥6 klinik sepsis başlangıcı sonrası."""
    pos = dev['SepsisLabel'].to_numpy() == 1
    h = dev.assign(p=pos.astype(int)).groupby('Patient_ID', sort=False)['p'].cumsum().to_numpy() - 1
    w = np.ones(len(dev))
    w[pos & (h < 6)] = early
    w[pos & (h >= 6)] = after
    return w


def load_features(dev, groups):
    mats, cols = [], []
    for name in groups:
        F, c = lab.features(dev, name, BUILDERS[name])
        mats.append(F)
        cols += c
    return (mats[0] if len(mats) == 1 else np.hstack(mats)), cols


def main():
    if len(sys.argv) > 2:                      # birden çok deney sırayla: deney.py a b c
        for n in sys.argv[1:]:
            sys.argv = [sys.argv[0], n]
            main()
        return
    if len(sys.argv) < 2 or sys.argv[1] == '--liste':
        results = lab.load_log()
        print('Tanımlı deneyler:')
        for name, (desc, *_rest) in DENEYLER.items():
            r = results.get(name)
            score = f"ROC-AUC {r['summary']['roc_auc']['mean']:.4f} ± {r['summary']['roc_auc']['std']:.4f}" if r else 'henüz koşulmadı'
            print(f'  {name:20s} {score:32s} {desc}')
        return

    name = sys.argv[1]
    if name not in DENEYLER:
        sys.exit(f'Bilinmeyen deney: {name}. Liste için: --liste')
    desc, groups, params = DENEYLER[name]
    params = dict(params)
    agirlik = params.pop('agirlik', None)
    monoton = params.pop('monoton', None)
    reference = params.pop('referans', 'baseline')

    t0 = time.time()
    lab.log(f'=== Deney: {name} — {desc}')
    dev = lab.load_dev()
    fold, inner = lab.fold_assignments(dev)
    y = dev['SepsisLabel'].to_numpy()
    lab.log(f'Geliştirme seti: {dev["Patient_ID"].nunique():,} hasta, {len(dev):,} saat (test hastaları hariç)')
    F, cols = load_features(dev, groups)

    weight = row_weights(dev, *agirlik) if agirlik else None
    monotone = monotone_vector(cols, monoton) if monoton else None
    if monotone:
        lab.log(f'Monoton kısıtlı sütunlar: {[c for c, m in zip(cols, monotone) if m]}')
    oof, folds = lab.lgbm_cv(F, y, fold, inner, params=params, weight=weight, monotone=monotone)
    lab.save_result(name, desc, folds, {'groups': groups, 'n_features': len(cols), 'minutes': (time.time() - t0) / 60})
    s = lab.summarize(folds)
    lab.log(f'\nSONUÇ {name}: ROC-AUC {s["roc_auc"]["mean"]:.4f} ± {s["roc_auc"]["std"]:.4f} · '
            f'PR-AUC {s["pr_auc"]["mean"]:.4f} ± {s["pr_auc"]["std"]:.4f} · {(time.time() - t0) / 60:.1f} dk')
    cmp = lab.compare(name, reference)
    if cmp:
        lab.log(f'{reference} ile karşılaştırma:\n' + cmp[0])


if __name__ == '__main__':
    main()
