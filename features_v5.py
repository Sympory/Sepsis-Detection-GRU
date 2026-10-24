"""
Sepsis Tahmin Sistemi — v5 ek özellik grupları
==============================================

v4'ün 281 özelliğine eklenecek aday gruplar. Her grup ayrı bir fonksiyondur; böylece deneylerde
tek tek eklenip katkısı ölçülür (scripts/v5/deney.py). İşe yaramayan grup modele girmez.

Kural (v4 ile aynı): her özellik yalnızca o saate kadarki veriden hesaplanır — pencereler geriye bakar,
"genişleyen" istatistikler yatıştan o saate kadardır. Geleceğe bakma testi: tests/test_features.py.
"""

import numpy as np
import pandas as pd

from features import CLINICAL_COLUMNS, STATIC_COLUMNS, TIME_COLUMN, clean
from features_v4 import TREND_LABS, TREND_VITALS

LAB_COLUMNS = [c for c in CLINICAL_COLUMNS if c not in ('HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2')]
SLOPE_COLUMNS = TREND_VITALS + ['Lactate', 'WBC', 'Creatinine', 'Platelets']

# Saatlik "anormal" tanımları (ileri doldurulmuş son değere göre)
ABNORMAL = {
    'hr100': ('HR', '>', 100), 'hr120': ('HR', '>', 120), 'temp38': ('Temp', '>', 38.0), 'temp36': ('Temp', '<', 36.0),
    'resp22': ('Resp', '>', 22), 'sbp100': ('SBP', '<', 100), 'map65': ('MAP', '<', 65), 'spo2_92': ('O2Sat', '<', 92),
}


def _prepare(df, group_col):
    """v4 ile aynı temizlik ve ileri doldurma. Dönüş: (df, hasta kimliği serisi, ileri doldurulmuş değerler)."""
    df = clean(df)
    for col in CLINICAL_COLUMNS + STATIC_COLUMNS + [TIME_COLUMN]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if group_col is None:
        df = df.assign(_pid=0)
        group_col = '_pid'
    gid = df[group_col]
    filled = df.groupby(gid, sort=False)[CLINICAL_COLUMNS].ffill()
    return df, gid, filled


def _rolling(series, gid, w, how):
    r = series.groupby(gid, sort=False).rolling(w, min_periods=1)
    return getattr(r, how)().reset_index(level=0, drop=True)


def _finish(out, index):
    return pd.DataFrame(out, index=index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


# ----------------------------------------------------------------------------
# A) Uzun pencereler: 48 ve 72 saat
# ----------------------------------------------------------------------------
def group_uzun(df, group_col=None):
    df, gid, F = _prepare(df, group_col)
    out = {}
    for c in TREND_VITALS + TREND_LABS:
        for w in (48, 72):
            out[f'{c}_mean{w}'] = _rolling(F[c], gid, w, 'mean')
            out[f'{c}_min{w}'] = _rolling(F[c], gid, w, 'min')
            out[f'{c}_max{w}'] = _rolling(F[c], gid, w, 'max')
        if c in TREND_VITALS:
            out[f'{c}_std48'] = _rolling(F[c], gid, 48, 'std')
    return _finish(out, df.index)


# ----------------------------------------------------------------------------
# B) Trend eğimi: son w saatteki noktalara en küçük kareler doğrusu (birim/saat)
# ----------------------------------------------------------------------------
def group_egim(df, group_col=None):
    df, gid, F = _prepare(df, group_col)
    t = df.groupby(gid, sort=False).cumcount().astype(float)
    out = {}
    for c in SLOPE_COLUMNS:
        x = F[c]
        valid = x.notna().astype(float)
        parts = pd.DataFrame({'n': valid, 't': t * valid, 'x': x.fillna(0), 'tx': t * x.fillna(0), 'tt': t * t * valid})
        for w in (6, 24):
            s = parts.groupby(gid, sort=False).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            den = s['n'] * s['tt'] - s['t'] ** 2
            slope = (s['n'] * s['tx'] - s['t'] * s['x']) / den.where(den > 1e-9)
            out[f'{c}_slope{w}'] = slope.where(s['n'] >= 3)
    return _finish(out, df.index)


# ----------------------------------------------------------------------------
# C) Eşik aşımı: son 6/24 saatte kaç saat anormal, kaç saattir kesintisiz anormal
# ----------------------------------------------------------------------------
def group_esik(df, group_col=None):
    df, gid, F = _prepare(df, group_col)
    out = {}
    for name, (c, op, thr) in ABNORMAL.items():
        cond = (F[c] > thr) if op == '>' else (F[c] < thr)
        cond = cond.astype(float).where(F[c].notna(), 0.0)
        for w in (6, 24):
            out[f'abn_{name}_h{w}'] = _rolling(cond, gid, w, 'sum')
        block = (cond != cond.groupby(gid, sort=False).shift()).astype(int).groupby(gid, sort=False).cumsum()
        out[f'abn_{name}_run'] = cond.groupby([gid, block], sort=False).cumsum() * cond
    sirs = sum(((F[c] > t) if o == '>' else (F[c] < t)).astype(float)
               for c, o, t in (('HR', '>', 90), ('Resp', '>', 20)))
    sirs = sirs + ((F['Temp'] > 38) | (F['Temp'] < 36)).astype(float) + ((F['WBC'] > 12) | (F['WBC'] < 4)).astype(float)
    sirs2 = (sirs >= 2).astype(float)
    for w in (6, 24):
        out[f'abn_sirs2_h{w}'] = _rolling(sirs2, gid, w, 'sum')
    return _finish(out, df.index)


# ----------------------------------------------------------------------------
# D) Hastanın kendi normaline göre sapma
# ----------------------------------------------------------------------------
def group_sapma(df, group_col=None):
    df, gid, F = _prepare(df, group_col)
    out = {}
    for c in TREND_VITALS:
        ex = F[c].groupby(gid, sort=False).expanding(min_periods=3)
        mean = ex.mean().reset_index(level=0, drop=True)
        std = ex.std().reset_index(level=0, drop=True)
        out[f'{c}_z_self'] = (F[c] - mean) / std.where(std > 1e-6)
        first = df[c].groupby(gid, sort=False).transform('first')  # ilk ölçülen (boş olmayan) değer
        # Yatıştaki ilk ölçüm: yalnızca ilk ölçüm zamanı geçtikten sonra bilinir (geleceğe bakmamak için)
        seen = df[c].notna().astype(int).groupby(gid, sort=False).cumsum() > 0
        out[f'{c}_vs_first'] = (F[c] - first).where(seen)
    for c in ('Lactate', 'WBC', 'Creatinine', 'Platelets'):
        first = df[c].groupby(gid, sort=False).transform('first')  # ilk ölçülen (boş olmayan) değer
        seen = df[c].notna().astype(int).groupby(gid, sort=False).cumsum() > 0
        out[f'{c}_vs_first'] = (F[c] - first).where(seen)
    return _finish(out, df.index)


# ----------------------------------------------------------------------------
# E) Ölçüm örüntüsü: test isteme yoğunluğu (klinisyenin kaygısının izi)
# ----------------------------------------------------------------------------
def group_olcum(df, group_col=None):
    df, gid, F = _prepare(df, group_col)
    labs_now = df[LAB_COLUMNS].notna().sum(axis=1).astype(float)
    any_lab = (labs_now > 0).astype(float)
    vitals_now = df[['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']].notna().sum(axis=1).astype(float)
    step = df.groupby(gid, sort=False).cumcount()
    last_lab = step.where(any_lab > 0).groupby(gid, sort=False).ffill()
    out = {
        'labs_now': labs_now,
        'labs_h6': _rolling(labs_now, gid, 6, 'sum'),
        'labs_h24': _rolling(labs_now, gid, 24, 'sum'),
        'lab_draws_h24': _rolling(any_lab, gid, 24, 'sum'),
        'since_any_lab': (step - last_lab).clip(upper=72).fillna(72),
        'vitals_h6': _rolling(vitals_now, gid, 6, 'sum'),
        'temp_checks_h24': _rolling(df['Temp'].notna().astype(float), gid, 24, 'sum'),
    }
    return _finish(out, df.index)


GROUPS = {'uzun': group_uzun, 'egim': group_egim, 'esik': group_esik, 'sapma': group_sapma, 'olcum': group_olcum}
