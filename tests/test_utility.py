"""PhysioNet 2019 Utility: vektörel hesap resmi tanımla aynı olmalı."""

import numpy as np

from evaluate_clinical import UtilityTable, patient_utility


def official_normalized(ys, alarms):
    obs = best = inaction = 0.0
    off = 0
    for lab in ys:
        n = len(lab)
        a = alarms[off:off + n]
        best_pred = np.zeros(n, dtype=bool)
        if lab.any():
            t = int(np.argmax(lab)) + 6
            best_pred[max(0, t - 12): min(t + 4, n)] = True
        obs += patient_utility(lab.astype(bool), a)
        best += patient_utility(lab.astype(bool), best_pred)
        inaction += patient_utility(lab.astype(bool), np.zeros(n, dtype=bool))
        off += n
    return (obs - inaction) / (best - inaction)


def test_vectorized_equals_official():
    rng = np.random.default_rng(3)
    ys, pids = [], []
    for i in range(60):
        n = int(rng.integers(8, 80))
        lab = np.zeros(n, dtype=int)
        if i % 3 == 0:
            lab[int(rng.integers(0, n)):] = 1
        ys.append(lab)
        pids.append(np.full(n, i))
    y, pid = np.concatenate(ys), np.concatenate(pids)
    table = UtilityTable(y, pid)
    for p in (0.0, 0.1, 0.5, 1.0):
        alarms = rng.random(len(y)) < p
        assert abs(table.score(alarms) - official_normalized(ys, alarms)) < 1e-12


def test_perfect_and_no_alarm():
    lab = np.array([0] * 20 + [1] * 10)
    table = UtilityTable(lab, np.zeros(len(lab)))
    assert table.score(np.zeros(len(lab), dtype=bool)) == 0.0
    t = 20 + 6
    best = np.zeros(len(lab), dtype=bool)
    best[t - 12:t + 4] = True
    assert abs(table.score(best) - 1.0) < 1e-12
