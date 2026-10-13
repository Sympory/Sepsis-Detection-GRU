"""
Sepsis Tahmin Sistemi — Tahmin Açıklaması
=========================================

"Bu skoru ne yükseltti?" sorusunu yanıtlar. LightGBM dalındaki her ağaçta karar
yolu izlenir; her bölünmede skorun değişimi o bölünmeyi yapan özelliğe yazılır
(Saabas yöntemi, gbm_numpy.contributions). 281 özelliğin katkısı klinik değişkene
göre toplanır (ör. Temp, Temp_max6, Temp_d6 → "Ateş") ve okunur cümleye çevrilir.

Katkılar log-odds cinsindendir; ekranda yaklaşık olasılık puanına çevrilir:
    Δp ≈ p · (1 − p) · Δlog-odds
Bu doğrusal bir yaklaşımdır; sıralama ve yön kesin, büyüklük yaklaşıktır.
"""

import numpy as np

from features import CLINICAL_COLUMNS

LABELS = {
    'HR': 'Nabız', 'O2Sat': 'SpO2', 'Temp': 'Ateş', 'SBP': 'Sistolik KB', 'MAP': 'MAP', 'DBP': 'Diyastolik KB',
    'Resp': 'Solunum', 'EtCO2': 'EtCO2', 'BaseExcess': 'Baz açığı', 'HCO3': 'HCO3', 'FiO2': 'Oksijen desteği (FiO2)',
    'pH': 'pH', 'PaCO2': 'PaCO2', 'SaO2': 'SaO2', 'AST': 'AST', 'BUN': 'BUN', 'Alkalinephos': 'ALP',
    'Calcium': 'Kalsiyum', 'Chloride': 'Klor', 'Creatinine': 'Kreatinin', 'Bilirubin_direct': 'Direkt bilirubin',
    'Glucose': 'Glukoz', 'Lactate': 'Laktat', 'Magnesium': 'Magnezyum', 'Phosphate': 'Fosfat',
    'Potassium': 'Potasyum', 'Bilirubin_total': 'Total bilirubin', 'TroponinI': 'Troponin I', 'Hct': 'Hematokrit',
    'Hgb': 'Hemoglobin', 'PTT': 'PTT', 'WBC': 'WBC', 'Fibrinogen': 'Fibrinojen', 'Platelets': 'Trombosit',
}
UNITS = {'HR': 'atım/dk', 'O2Sat': '%', 'Temp': '°C', 'SBP': 'mmHg', 'MAP': 'mmHg', 'DBP': 'mmHg', 'Resp': '/dk',
         'Lactate': 'mmol/L', 'WBC': '10³/µL', 'Creatinine': 'mg/dL', 'Platelets': '10³/µL', 'BUN': 'mg/dL',
         'Glucose': 'mg/dL', 'Bilirubin_total': 'mg/dL'}
DERIVED = {
    'shock_index': ('shock', 'Şok indeksi (nabız / sistolik KB)'),
    'pulse_pressure': ('pulse_pressure', 'Nabız basıncı'),
    'bun_creatinine': ('bun_cr', 'BUN / kreatinin oranı'),
    'spo2_fio2': ('spo2_fio2', 'SpO2 / FiO2 oranı'),
    'sirs': ('sirs', 'SIRS kriterleri'),
    'qsofa_partial': ('qsofa', 'qSOFA (solunum, tansiyon)'),
    'sofa_partial': ('sofa', 'Kısmi SOFA skoru'),
    'sofa_partial_d24': ('sofa', 'Kısmi SOFA skoru'),
    'lactate_high': (('Lactate', 'value'), None),
    'Hour': ('icu_time', "YBÜ'de geçen süre"),
    'n_hours': ('icu_time', "YBÜ'de geçen süre"),
    'Age': ('age', 'Yaş'),
    'Gender': ('gender', 'Cinsiyet'),
}
_BY_LENGTH = sorted(CLINICAL_COLUMNS, key=len, reverse=True)


def group_of(name):
    """Özellik adı → (grup anahtarı, ekran başlığı)"""
    if name in DERIVED:
        key, label = DERIVED[name]
        if isinstance(key, tuple):
            return key, LABELS[key[0]]
        return key, label
    for var in _BY_LENGTH:
        if name == var or name.startswith(var + '_'):
            suffix = name[len(var) + 1:]
            if suffix in ('measured', 'since', 'count'):
                return (var, 'measure'), f'{LABELS[var]} ölçümü'
            return (var, 'value'), LABELS[var]
    return name, name


def _fmt(v, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    v = float(v)
    return f'{v:.0f}' if abs(v) >= 100 or float(v).is_integer() else f'{v:.{digits}f}'


def _describe(key, row):
    """Grubun o anki değerini kısa cümleyle anlatır."""
    g = row.get
    if isinstance(key, tuple):
        var, kind = key
        if kind == 'measure':
            since, count = g(f'{var}_since'), g(f'{var}_count')
            if not count:
                return 'hiç ölçülmedi'
            return 'bu saat ölçüldü' if since == 0 else f'son ölçüm {int(since)} saat önce'
        value = g(var)
        if value is None or np.isnan(value):
            return 'ölçülmedi'
        scale, unit = (100, '%') if var == 'FiO2' else (1, UNITS.get(var, ''))
        text = f'{_fmt(value * scale)} {unit}'.strip()
        d6 = g(f'{var}_d6')
        if d6 is not None and not np.isnan(d6) and abs(d6) > 1e-9:
            text += f' (6 saatte {"+" if d6 > 0 else ""}{_fmt(d6 * scale)})'
        return text
    if key == 'icu_time':
        return f'{int(g("Hour", 0))}. saat'
    if key == 'shock':
        return _fmt(g('shock_index'), 2)
    if key == 'sirs':
        return f'{_fmt(g("sirs"))} / 4'
    if key == 'qsofa':
        return f'{_fmt(g("qsofa_partial"))} / 2'
    if key == 'sofa':
        return _fmt(g('sofa_partial'))
    if key == 'age':
        return _fmt(g('Age'))
    if key == 'gender':
        return {1.0: 'erkek', 0.0: 'kadın'}.get(g('Gender'), '—')
    if key == 'spo2_fio2':
        return _fmt(g('spo2_fio2'), 0)
    if key == 'pulse_pressure':
        return f'{_fmt(g("pulse_pressure"))} mmHg'
    if key == 'bun_cr':
        return _fmt(g('bun_creatinine'))
    return ''


def explain(gbm, feature_row, top_up=3, top_down=2):
    """
    feature_row: tek satırlık pandas Series/DataFrame satırı (features_v4 çıktısı).
    Dönüş: {'method', 'factors_up', 'factors_down'}; her faktör {label, detail, points}.
    """
    names = gbm.feature_names
    row = {n: float(v) for n, v in zip(names, np.asarray(feature_row, dtype=float).ravel())}
    contrib, bias = gbm.contributions(np.array([row[n] for n in names]))
    raw = bias + contrib.sum()
    p = 1.0 / (1.0 + np.exp(-raw))
    scale = p * (1 - p) * 100   # log-odds → yaklaşık yüzde puan

    groups = {}
    for name, c in zip(names, contrib):
        if c == 0:
            continue
        key, label = group_of(name)
        k = str(key)
        if k not in groups:
            groups[k] = {'key': key, 'label': label, 'logit': 0.0}
        groups[k]['logit'] += c

    factors = []
    for gdata in groups.values():
        factors.append({'label': gdata['label'], 'detail': _describe(gdata['key'], row),
                        'points': round(gdata['logit'] * scale, 1)})
    up = sorted([f for f in factors if f['points'] >= 0.1], key=lambda f: -f['points'])[:top_up]
    down = sorted([f for f in factors if f['points'] <= -0.1], key=lambda f: f['points'])[:top_down]
    return {'method': 'Saabas katkıları (LightGBM dalı), yaklaşık yüzde puan',
            'baseline_probability': round(float(1 / (1 + np.exp(-bias))), 4),
            'factors_up': up, 'factors_down': down}
