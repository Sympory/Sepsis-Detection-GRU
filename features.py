"""
Sepsis Tahmin Sistemi — v3 Özellik Mühendisliği
===============================================

Eğitim (scripts/train_v3.py) ve web uygulaması (app.py) AYNI fonksiyonu
kullanır; böylece eğitimde görülen girdi ile canlı girdi birebir aynıdır.

Eski pipeline'ın sorunu: eksik değerler doğrudan medyanla dolduruluyordu.
PhysioNet verisinde laboratuvar değerlerinin %88-99'u eksik olduğu için
model değerin kendisini değil "medyandan farklı mı = ölçülmüş mü" bilgisini
öğreniyordu (ör. düşük laktat riski artırıyordu).

v3 yaklaşımı (PhysioNet 2019 literatüründeki standart):
  1. Hasta içinde ileri doldurma (son ölçülen değer geçerli kabul edilir)
  2. Her klinik değişken için ölçüm maskesi (bu saat ölçüldü mü: 0/1)
  3. Kalan eksikler eğitim medyanıyla, sonra standartlaştırma
  4. 6 saatten kısa geçmiş: ilk satır geriye kopyalanır (uygulamayla aynı)
"""

import numpy as np
import pandas as pd

CLINICAL_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets',
]
# HospAdmTime bilerek çıkarıldı: Yaş + Cinsiyet + HospAdmTime hastaların %93'ünü
# tekil olarak tanımlıyor; model bunu kullanarak hastayı "ezberliyordu"
# (eğitim AUC 0.92, validasyon 0.75).
STATIC_COLUMNS = ['Age', 'Gender']
TIME_COLUMN = 'Hour'

# Fizyolojik olarak imkânsız değerler NaN yapılır (eğitim verisi temizliği)
PHYSIOLOGIC_LIMITS = {
    'HR': (0, 300), 'MAP': (0, 250), 'SBP': (0, 300), 'DBP': (0, 250),
    'Temp': (25, 45), 'O2Sat': (0, 100), 'SaO2': (0, 100), 'Resp': (0, 100),
    'FiO2': (0, 1.0), 'pH': (6.5, 8.0),
}


def feature_names():
    return ([*CLINICAL_COLUMNS]
            + [f'{c}_measured' for c in CLINICAL_COLUMNS]
            + STATIC_COLUMNS + [TIME_COLUMN])


def clean(df):
    df = df.copy()
    for col, (lo, hi) in PHYSIOLOGIC_LIMITS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def build_hourly_features(df, group_col=None):
    """
    Saatlik satırlardan (zaman sırasına dizilmiş) ham özellik tablosu üretir.
    Henüz medyan doldurma ve ölçekleme yapılmaz.
    group_col verilirse ileri doldurma her hasta içinde ayrı yapılır.
    """
    df = clean(df)
    for col in CLINICAL_COLUMNS + STATIC_COLUMNS + [TIME_COLUMN]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    masks = df[CLINICAL_COLUMNS].notna().astype(np.float32)
    masks.columns = [f'{c}_measured' for c in CLINICAL_COLUMNS]

    if group_col:
        filled = df.groupby(group_col, sort=False)[CLINICAL_COLUMNS + STATIC_COLUMNS].ffill()
    else:
        filled = df[CLINICAL_COLUMNS + STATIC_COLUMNS].ffill()

    out = pd.concat([filled[CLINICAL_COLUMNS], masks, filled[STATIC_COLUMNS], df[[TIME_COLUMN]]], axis=1)
    return out[feature_names()]


class FeatureTransformer:
    """Medyan doldurma + standartlaştırma. Yalnızca eğitim hastalarıyla fit edilir."""

    def __init__(self):
        self.medians = None
        self.means = None
        self.stds = None
        self.names = feature_names()

    def fit(self, feats):
        values = feats[self.names].to_numpy(dtype=np.float64)
        # Hiç ölçülmemiş sütunun medyanı NaN olur (ör. EtCO2) → 0
        self.medians = np.nan_to_num(np.nanmedian(values, axis=0), nan=0.0)
        filled = np.where(np.isnan(values), self.medians, values)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        # Maskeler 0/1 kalsın (ölçeklenmez)
        for i, n in enumerate(self.names):
            if n.endswith('_measured'):
                self.means[i], self.stds[i] = 0.0, 1.0
        return self

    def transform(self, feats):
        values = feats[self.names].to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values), self.medians, values)
        return ((values - self.means) / self.stds).astype(np.float32)

    def clinical_stats(self):
        """Güvenilirlik modülünün dağılım dışılık kontrolü için {sütun: (ortalama, std)}"""
        return {n: (float(self.means[i]), float(self.stds[i]))
                for i, n in enumerate(self.names) if n in CLINICAL_COLUMNS}


def make_windows(X, window=6, pad_short=True):
    """
    Tek hastanın (T, F) matrisinden her saat için (window, F) pencere üretir.
    pad_short=True ise ilk saatler için ilk satır geriye kopyalanır
    (uygulama 1. saatten itibaren tahmin yaptığı için eğitim de bunu görmeli).
    """
    T = len(X)
    if T == 0:
        return np.empty((0, window, X.shape[1]), dtype=np.float32)
    if pad_short:
        padded = np.concatenate([np.repeat(X[:1], window - 1, axis=0), X], axis=0)
        idx = np.arange(T)[:, None] + np.arange(window)[None, :]
        return padded[idx]
    if T < window:
        return np.empty((0, window, X.shape[1]), dtype=np.float32)
    idx = np.arange(T - window + 1)[:, None] + np.arange(window)[None, :]
    return X[idx]
