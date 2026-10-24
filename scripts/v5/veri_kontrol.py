"""
Veri bütünlüğü kontrolü: data/train.csv orijinal PhysioNet 2019 verisi mi, bozulmuş/değiştirilmiş mi?

Kontroller:
  1. Satır, hasta, sepsis sayıları resmi PhysioNet 2019 değerleriyle aynı mı?
  2. Sütunlar: resmi 40 sütun dışında (ör. sentetik biyobelirteç) sütun var mı?
  3. Tekrarlayan satır, eksik/sırasız saat var mı?
  4. Etiket tutarlı mı? (bir kez 1 olunca 0'a dönmemeli)
  5. Eksik değer oranları ham veriye uygun mu? (doldurulmuş/"optimize" edilmiş veride eksik kalmaz)
  6. Fizyolojik olarak imkânsız değer var mı?
  7. Yaş ve cinsiyet hasta içinde sabit mi?
  8. Hastane dağılımı (A: 20.336, B: 20.000)

  py -3.10 scripts/v5/veri_kontrol.py
"""

import os
import sys

import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Resmi PhysioNet 2019 eğitim verisi (training_setA + training_setB)
OFFICIAL = {'patients': 40336, 'rows': 1552210, 'sepsis_patients': 2932, 'positive_rows': 27916,
            'hospital_A': 20336, 'hospital_B': 20000}
OFFICIAL_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH',
                    'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
                    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
                    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
# Fizyolojik olarak mümkün aralıklar (bunun dışı = veri hatası şüphesi)
PLAUSIBLE = {'HR': (10, 300), 'O2Sat': (20, 100), 'Temp': (20, 45), 'SBP': (20, 300), 'MAP': (10, 300),
             'DBP': (5, 300), 'Resp': (1, 100), 'pH': (6.5, 8.0), 'Lactate': (0, 40), 'WBC': (0, 500),
             'Platelets': (0, 3000), 'Creatinine': (0, 50), 'FiO2': (0.1, 1.0), 'Age': (14, 110)}

problems, notes = [], []


def check(ok, text):
    print(f"  {'✓' if ok else '✗'} {text}")
    if not ok:
        problems.append(text)


def main():
    path = os.path.join(ROOT, 'data', 'train.csv')
    print(f'Dosya: {path} ({os.path.getsize(path) / 1e6:.0f} MB)')
    df = pd.read_csv(path)
    extra = [c for c in df.columns if c not in OFFICIAL_COLUMNS]
    print(f'Resmi sütunlar dışındakiler: {extra}')

    print('\n1. Sayılar (resmi PhysioNet 2019 ile)')
    status = df.groupby('Patient_ID')['SepsisLabel'].max()
    got = {'patients': df['Patient_ID'].nunique(), 'rows': len(df), 'sepsis_patients': int(status.sum()),
           'positive_rows': int(df['SepsisLabel'].sum()),
           'hospital_A': int((status.index < 100000).sum()), 'hospital_B': int((status.index >= 100000).sum())}
    for k, v in OFFICIAL.items():
        check(got[k] == v, f'{k:16s} {got[k]:>10,}  (resmi {v:,})')

    print('\n2. Sütunlar')
    missing_cols = [c for c in OFFICIAL_COLUMNS if c not in df.columns]
    check(not missing_cols, f'Resmi 41 sütunun hepsi var' + (f' — eksik: {missing_cols}' if missing_cols else ''))
    suspicious = [c for c in extra if c not in ('Unnamed: 0', 'Hour', 'Patient_ID')]
    check(not suspicious, 'Fazladan klinik/sentetik sütun yok' + (f' — bulundu: {suspicious}' if suspicious else ''))

    print('\n3. Saat düzeni')
    dup = df.duplicated(['Patient_ID', 'ICULOS']).sum()
    check(dup == 0, f'Aynı hasta-saat tekrarı: {dup}')
    g = df.groupby('Patient_ID', sort=False)
    steps = g['ICULOS'].diff().dropna()
    check((steps == 1).all(), f'ICULOS her satırda 1 artıyor (aykırı adım: {(steps != 1).sum()})')
    hour_ok = (df['Hour'] == g.cumcount()).all() if 'Hour' in df else True
    check(hour_ok, 'Hour sütunu hasta içi sıra numarasıyla aynı (0, 1, 2…)')
    start = g['ICULOS'].first()
    notes.append(f'ICULOS 1\'den başlayan hasta oranı: %{(start == 1).mean() * 100:.1f} '
                 f'(diğerleri kayda geç başlamış; ham veride de böyle)')

    print('\n4. Etiket tutarlılığı')
    back = (g['SepsisLabel'].diff() < 0).sum()
    check(back == 0, f'Etiket 1 → 0 dönüşü: {back} (sepsis başlayınca etiket kalıcı olmalı)')
    check(df['SepsisLabel'].isna().sum() == 0, f'Boş etiket: {df["SepsisLabel"].isna().sum()}')
    first_pos = df[df['SepsisLabel'] == 1].groupby('Patient_ID')['ICULOS'].min()
    notes.append(f'Sepsis etiketinin başladığı saat (medyan ICULOS): {first_pos.median():.0f}; '
                 f'ilk saatte etiketli hasta: {(first_pos <= start.loc[first_pos.index]).sum()}')

    print('\n5. Eksik değer oranları (ham veride laboratuvarların çoğu boş olmalı)')
    miss = df[['HR', 'Temp', 'MAP', 'Lactate', 'WBC', 'pH', 'Bilirubin_direct', 'TroponinI']].isna().mean() * 100
    print('     ' + '  '.join(f'{k} %{v:.0f}' for k, v in miss.items()))
    check(miss['Lactate'] > 90 and miss['HR'] > 3, 'Eksik değerler doldurulmamış (laktat > %90 boş, HR'
                                                    ' de kısmen boş) → ham veri')

    print('\n6. Fizyolojik aralık dışı değerler')
    for c, (lo, hi) in PLAUSIBLE.items():
        v = df[c].dropna()
        bad = ((v < lo) | (v > hi)).sum()
        if bad:
            notes.append(f'{c}: {bad} değer [{lo}, {hi}] dışında (min {v.min():g}, maks {v.max():g})')
    check(not any('dışında' in n for n in notes), 'Aralık dışı değer yok' if True else '')

    print('\n7. Hasta içinde sabit olması gerekenler')
    for c in ('Age', 'Gender'):
        varying = (g[c].nunique() > 1).sum()
        check(varying == 0, f'{c} hasta içinde değişen hasta sayısı: {varying}')

    print('\nNotlar:')
    for n in notes:
        print('  · ' + n)
    print(f"\nSONUÇ: {'SORUN YOK — veri orijinal PhysioNet 2019 ile tutarlı.' if not problems else str(len(problems)) + ' kontrol başarısız:'}")
    for p in problems:
        print('  ✗ ' + p)


if __name__ == '__main__':
    main()
