"""Özellik mühendisliği testleri: geleceğe bakmama, ileri doldurma, ölçüm süresi, giriş normalizasyonu."""

import numpy as np
import pandas as pd
import pytest

from features_v4 import V4Scaler, build_v4, feature_names_v4
from reliability import normalize_inputs


def patient(hours, overrides=None):
    overrides = overrides or {}
    rows = []
    for h in range(hours):
        r = {'Hour': h, 'HR': 80 + h, 'Temp': 37.0, 'Resp': 16, 'SBP': 120, 'O2Sat': 98, 'Age': 60, 'Gender': 1}
        r.update(overrides.get(h, {}))
        rows.append(r)
    return pd.DataFrame(rows)


def test_feature_count():
    assert len(feature_names_v4()) == 281


def test_no_future_leakage():
    """t. saatin özellikleri, sonraki saatler eklenince değişmemeli."""
    short = build_v4(patient(10))
    long = build_v4(patient(30, {h: {'HR': 160, 'Temp': 40.0, 'Lactate': 8.0} for h in range(10, 30)}))
    pd.testing.assert_frame_equal(short, long.iloc[:10])


def test_forward_fill_and_since():
    df = patient(6, {1: {'Lactate': 3.0}})
    f = build_v4(df)
    assert f['Lactate'].tolist()[1:] == [3.0] * 5                 # ileri doldurma
    assert np.isnan(f['Lactate'].iloc[0])
    assert f['Lactate_since'].tolist() == [72, 0, 1, 2, 3, 4]     # son ölçümden geçen saat (72 = hiç)
    assert f['Lactate_count'].tolist() == [0, 1, 1, 1, 1, 1]
    assert f['Lactate_measured'].tolist() == [0, 1, 0, 0, 0, 0]


def test_patients_do_not_mix():
    a = patient(4).assign(Patient_ID=1)
    b = patient(4, {0: {'Lactate': 9.0}}).assign(Patient_ID=2)
    f = build_v4(pd.concat([a, b], ignore_index=True), group_col='Patient_ID')
    assert f['Lactate'].iloc[:4].isna().all(), "Bir hastanın değeri diğerine taşınmamalı"


def test_trend_and_derived():
    f = build_v4(patient(8))
    assert f['HR_d6'].iloc[7] == pytest.approx(6.0)               # 87 − 81
    assert f['HR_max24'].iloc[7] == 87
    assert f['shock_index'].iloc[0] == pytest.approx(80 / 120)


def test_scaler_handles_missing():
    f = build_v4(patient(5))
    X = V4Scaler().fit(f).transform(f)
    assert np.isfinite(X).all() and X.shape == (5, 281)


def test_normalize_inputs_fio2_and_map():
    vs, notes = normalize_inputs({'FiO2': 40, 'SBP': 120, 'DBP': 60})
    assert vs['FiO2'] == pytest.approx(0.40)
    assert vs['MAP'] == pytest.approx(80.0)
    assert len(notes) == 2


@pytest.mark.parametrize('group', ['uzun', 'egim', 'esik', 'sapma', 'olcum'])
def test_v5_groups_no_future_leakage(group):
    """v5 ek grupları: t. saatin özellikleri sonraki saatler eklenince değişmemeli."""
    from features_v5 import GROUPS
    build = GROUPS[group]
    short = build(patient(10, {3: {'Lactate': 2.0}}))
    long = build(patient(40, {3: {'Lactate': 2.0}, **{h: {'HR': 150, 'Temp': 40.0, 'Lactate': 8.0} for h in range(10, 40)}}))
    pd.testing.assert_frame_equal(short, long.iloc[:10])


def test_v5_slope_and_run_length():
    from features_v5 import group_egim, group_esik
    df = patient(8)                                   # HR 80, 81, …, 87 → eğim 1/saat
    assert group_egim(df)['HR_slope6'].iloc[7] == pytest.approx(1.0)
    hot = patient(6, {h: {'Temp': 38.6} for h in (2, 3, 4)})
    run = group_esik(hot)['abn_temp38_run'].tolist()
    assert run == [0, 0, 1, 2, 3, 0]
