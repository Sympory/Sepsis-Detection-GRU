"""
NumPy ağaç değerlendiricisi testleri.

LightGBM bu süreçte yüklenmez (TensorFlow ile DLL çakışması). LightGBM ile birebir
eşleşme scripts/train_hybrid.py ve scripts/enrich_trees.py içinde doğrulanır; burada
elle kurulmuş bir ağaçla eksik değer kuralları ve açıklama özdeşliği test edilir.
"""

import os

import numpy as np
import pytest

from gbm_numpy import NumpyGBM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# x0 <= 1 ? (x1 <= 5 ? 0.1 : 0.3) : -0.2   — kök: NaN → sol (NaN türü), x1 düğümü: None türü (NaN → 0)
TREE = {'feature_names': ['x0', 'x1'], 'trees': [{
    'feat': [0, 1], 'thr': [1.0, 5.0], 'dleft': [True, False], 'mtype': [2, 0],
    'left': [1, ~0], 'right': [~2, ~1], 'leaf': [0.1, 0.3, -0.2], 'value': [0.05, 0.15], 'root': 0}]}


@pytest.mark.parametrize('x,expected', [
    ([0.5, 3.0], 0.1),           # sol, sol
    ([0.5, 9.0], 0.3),           # sol, sağ
    ([2.0, 3.0], -0.2),          # sağ
    ([np.nan, 9.0], 0.3),        # NaN türü: varsayılan sol
    ([0.5, np.nan], 0.1),        # None türü: NaN → 0 → 0 <= 5 sol
])
def test_tree_routing(x, expected):
    m = NumpyGBM(TREE)
    assert m.predict_raw(np.array([x]))[0] == pytest.approx(expected)


def test_contributions_sum_to_raw():
    m = NumpyGBM(TREE)
    for x in ([0.5, 3.0], [0.5, 9.0], [2.0, 3.0]):
        c, b = m.contributions(np.array(x))
        assert c.sum() + b == pytest.approx(m.predict_raw(np.array([x]))[0])


def test_deployed_model_contributions():
    path = os.path.join(ROOT, 'models', 'v4', 'trees.json')
    if not os.path.exists(path):
        pytest.skip('v4 modeli yok')
    gbm = NumpyGBM.load(path)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, len(gbm.feature_names)))
    X[rng.random(X.shape) < 0.4] = np.nan
    for x in X:
        c, b = gbm.contributions(x)
        raw = np.mean([m.predict_raw(x[None])[0] for m in gbm.members])
        assert c.sum() + b == pytest.approx(raw, abs=1e-6)
