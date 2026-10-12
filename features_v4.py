"""
Sepsis Tahmin Sistemi — v4 Zaman Serisi Özellikleri
===================================================

v3'ün (ileri doldurma + ölçüm maskeleri) üzerine, her saat için hastanın
geçmişinden hesaplanan özetler eklenir. Model "nabız şu an 110" yerine
"nabız 24 saattir yükseliyor, ortalaması 95, en yükseği 118" bilgisini görür.

Tüm özellikler yalnızca o saate kadarki veriden hesaplanır (geleceğe bakmaz),
dolayısıyla hem eğitimde hem uygulamada aynı fonksiyon kullanılabilir.
"""

import numpy as np
import pandas as pd

from features import CLINICAL_COLUMNS, STATIC_COLUMNS, TIME_COLUMN, clean

TREND_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']
TREND_LABS = ['Lactate', 'WBC', 'Creatinine', 'Platelets', 'BUN', 'FiO2']
WINDOWS = (6, 24)
MAX_GAP = 72   # "son ölçümden bu yana" üst sınırı (saat)


def build_v4(df, group_col=None):
    """
    df: saatlik satırlar (hasta içinde zaman sırasında). group_col verilirse hasta bazlı hesaplanır.
    Dönüş: özellik DataFrame'i (NaN'lar korunur; ağaç modelleri NaN'ı kendisi işler).
    """
    df = clean(df)
    for col in CLINICAL_COLUMNS + STATIC_COLUMNS + [TIME_COLUMN]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if group_col is None:
        df = df.assign(_pid=0)
        group_col = '_pid'
    g = df.groupby(group_col, sort=False)

    out = {}
    measured = df[CLINICAL_COLUMNS].notna()
    filled = g[CLINICAL_COLUMNS].ffill()

    # 1) Son bilinen değerler ve bu saat ölçüldü mü
    for c in CLINICAL_COLUMNS:
        out[c] = filled[c]
        out[f'{c}_measured'] = measured[c].astype(np.float32)

    # 2) Son ölçümden bu yana geçen saat ve şimdiye kadarki ölçüm sayısı
    step = g.cumcount()
    for c in CLINICAL_COLUMNS:
        last_idx = step.where(measured[c]).groupby(df[group_col]).ffill()
        out[f'{c}_since'] = (step - last_idx).clip(upper=MAX_GAP).fillna(MAX_GAP).astype(np.float32)
        out[f'{c}_count'] = measured[c].astype(np.int32).groupby(df[group_col]).cumsum().astype(np.float32)

    # 3) Eğilimler: pencere istatistikleri ve değişimler (ileri doldurulmuş değer üzerinden)
    fg = filled.groupby(df[group_col])
    for c in TREND_VITALS + TREND_LABS:
        for w in WINDOWS:
            roll = fg[c].rolling(w, min_periods=1)
            out[f'{c}_mean{w}'] = roll.mean().reset_index(level=0, drop=True)
            out[f'{c}_min{w}'] = roll.min().reset_index(level=0, drop=True)
            out[f'{c}_max{w}'] = roll.max().reset_index(level=0, drop=True)
            if c in TREND_VITALS:
                out[f'{c}_std{w}'] = roll.std().reset_index(level=0, drop=True)
        out[f'{c}_d1'] = filled[c] - fg[c].shift(1)
        out[f'{c}_d6'] = filled[c] - fg[c].shift(6)

    # 3b) Yatıştan bu yana (genişleyen pencere) en kötü değerler ve şimdiki değerin bunlara uzaklığı
    for c in TREND_VITALS:
        ex = fg[c].expanding(min_periods=1)
        mx = ex.max().reset_index(level=0, drop=True)
        mn = ex.min().reset_index(level=0, drop=True)
        out[f'{c}_maxall'] = mx
        out[f'{c}_minall'] = mn
        out[f'{c}_meanall'] = ex.mean().reset_index(level=0, drop=True)
        out[f'{c}_vs_mean'] = filled[c] - out[f'{c}_meanall']

    # 4) Klinik türetilmiş skorlar
    F = filled
    out['shock_index'] = F['HR'] / F['SBP']
    out['pulse_pressure'] = F['SBP'] - F['DBP']
    out['bun_creatinine'] = F['BUN'] / F['Creatinine']
    fio2 = F['FiO2'].where(F['FiO2'] > 0)
    out['spo2_fio2'] = F['O2Sat'] / fio2
    sirs = ((F['HR'] > 90).astype(int) + ((F['Temp'] > 38) | (F['Temp'] < 36)).astype(int)
            + (F['Resp'] > 20).astype(int) + ((F['WBC'] > 12) | (F['WBC'] < 4)).astype(int))
    out['sirs'] = sirs.astype(np.float32)
    out['qsofa_partial'] = ((F['Resp'] >= 22).astype(int) + (F['SBP'] <= 100).astype(int)).astype(np.float32)
    # Kısmi SOFA: ölçülen bileşenlerin toplamı (hiçbiri yoksa NaN)
    sofa = pd.concat([
        pd.cut(F['Platelets'], [-np.inf, 20, 50, 100, 150, np.inf], labels=[4, 3, 2, 1, 0]).astype(float),
        pd.cut(F['Bilirubin_total'], [-np.inf, 1.2, 2, 6, 12, np.inf], labels=[0, 1, 2, 3, 4]).astype(float),
        (F['MAP'] < 70).astype(float).where(F['MAP'].notna()),
        pd.cut(F['Creatinine'], [-np.inf, 1.2, 2, 3.5, 5, np.inf], labels=[0, 1, 2, 3, 4]).astype(float),
    ], axis=1).sum(axis=1, min_count=1)
    out['sofa_partial'] = sofa
    out['sofa_partial_d24'] = sofa - sofa.groupby(df[group_col]).shift(24)
    out['lactate_high'] = (F['Lactate'] > 2).astype(np.float32).where(F['Lactate'].notna())

    # 5) Statik ve zaman
    for c in STATIC_COLUMNS:
        out[c] = g[c].ffill()
    out['Hour'] = df[TIME_COLUMN]
    out['n_hours'] = (step + 1).astype(np.float32)

    feats = pd.DataFrame(out, index=df.index)
    return feats.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def feature_names_v4():
    demo = pd.DataFrame({'HR': [80.0], 'Hour': [0]})
    return list(build_v4(demo).columns)


class V4Scaler:
    """Sinir ağı girdisi için: medyan doldurma + standartlaştırma + kırpma. Yalnızca eğitim hastalarıyla fit edilir."""

    def fit(self, F):
        self.columns = list(F.columns)
        self.median = F.median().fillna(0)
        filled = F.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std().replace(0, 1).fillna(1)
        return self

    def transform(self, F):
        Z = (F[self.columns].fillna(self.median) - self.mean) / self.std
        return Z.clip(-6, 6).to_numpy(np.float32)

    def clinical_stats(self):
        """Güvenilirlik modülünün dağılım dışılık kontrolü için {sütun: (ortalama, std)}"""
        return {c: (float(self.mean[c]), float(self.std[c])) for c in CLINICAL_COLUMNS if c in self.mean.index}

