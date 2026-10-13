"""
LightGBM ağaçlarını saf NumPy ile çalıştırır.

Neden: Windows'ta LightGBM'in C kütüphanesi TensorFlow ile aynı süreçte
bazı fonksiyonlarda çöküyor (access violation). Uygulama TensorFlow'u zaten
yüklediği için, eğitilmiş ağaçlar JSON'a aktarılır ve tahmin bu modülde
yalnızca NumPy ile yapılır. Sonuç LightGBM'in kendi tahminiyle birebir aynıdır
(scripts/train_hybrid.py bunu doğrular).

  export_lightgbm(booster, path)   → yalnızca eğitim sürecinde (TensorFlow yokken)
  NumpyGBM.load(path).predict(X)   → uygulamada, LightGBM kurulu olmasa bile çalışır
"""

import json

import numpy as np

MISSING = {'None': 0, 'Zero': 1, 'NaN': 2}
ZERO_THRESHOLD = 1e-35


def _flatten(tree):
    """İç içe ağaç yapısını dizilere çevirir. Çocuk < 0 ise yaprak: ~çocuk = yaprak indeksi."""
    feat, thr, dleft, mtype, left, right, leaves, values = [], [], [], [], [], [], [], []

    def visit(node):
        if 'leaf_value' in node:
            leaves.append(float(node['leaf_value']))
            return ~(len(leaves) - 1)
        if node.get('decision_type', '<=') != '<=':
            raise ValueError('Yalnızca sayısal (<=) bölünmeler desteklenir')
        i = len(feat)
        feat.append(int(node['split_feature']))
        thr.append(float(node['threshold']))
        dleft.append(bool(node['default_left']))
        mtype.append(MISSING[node.get('missing_type', 'None')])
        values.append(float(node.get('internal_value', 0.0)))   # düğümün ortalama çıktısı (açıklama için)
        left.append(0)
        right.append(0)
        left[i] = visit(node['left_child'])
        right[i] = visit(node['right_child'])
        return i

    root = visit(tree['tree_structure'])
    return {'feat': feat, 'thr': thr, 'dleft': dleft, 'mtype': mtype,
            'left': left, 'right': right, 'leaf': leaves, 'value': values, 'root': root}


def _export(booster):
    dump = booster.dump_model()
    return {'feature_names': dump['feature_names'], 'objective': dump.get('objective', ''),
            'trees': [_flatten(t) for t in dump['tree_info']]}


def export_lightgbm(boosters, path):
    """Bir veya birden çok LightGBM modelini (seed topluluğu) tek JSON dosyasına yazar."""
    boosters = boosters if isinstance(boosters, (list, tuple)) else [boosters]
    model = {'models': [_export(b) for b in boosters]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model, f)
    return model


class NumpyGBM:
    def __init__(self, model):
        self.feature_names = model['feature_names']
        self.trees = []
        for t in model['trees']:
            self.trees.append({k: np.asarray(v) for k, v in t.items() if k != 'root'} | {'root': t['root']})

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return NumpyGBMEnsemble([cls(m) for m in data['models']]) if 'models' in data else cls(data)

    def _tree_raw(self, t, X):
        n = len(X)
        if t['root'] < 0:                       # tek yapraklı ağaç
            return np.full(n, t['leaf'][~t['root']])
        out = np.empty(n)
        node = np.full(n, t['root'])
        active = np.arange(n)
        while active.size:
            nd = node[active]
            x = X[active, t['feat'][nd]]
            mt = t['mtype'][nd]
            nan = np.isnan(x)
            x = np.where(nan & (mt == 0), 0.0, x)                       # None: NaN → 0
            missing = ((mt == 2) & nan) | ((mt == 1) & (nan | (np.abs(x) <= ZERO_THRESHOLD)))
            go_left = np.where(missing, t['dleft'][nd], x <= t['thr'][nd])
            child = np.where(go_left, t['left'][nd], t['right'][nd])
            leaf = child < 0
            out[active[leaf]] = t['leaf'][~child[leaf]]
            node[active[~leaf]] = child[~leaf]
            active = active[~leaf]
        return out

    def predict_raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        return sum(self._tree_raw(t, X) for t in self.trees)

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-self.predict_raw(X)))

    def contributions(self, x):
        """
        Tek satır için özellik katkıları (log-odds), Saabas yöntemi: karar yolundaki her
        bölünmede çocuk düğümün değeri ile düğümün değeri arasındaki fark, bölünmeyi yapan
        özelliğe yazılır. Dönüş: (katkı dizisi, taban değer); taban + katkılar = ham skor.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        contrib = np.zeros(len(self.feature_names))
        bias = 0.0
        for t in self.trees:
            if 'value' not in t:
                raise ValueError('Ağaç dosyasında düğüm değerleri yok (eski dışa aktarım)')
            node = t['root']
            if node < 0:
                bias += t['leaf'][~node]
                continue
            bias += t['value'][node]
            while node >= 0:
                f = t['feat'][node]
                v = x[f]
                mt = t['mtype'][node]
                nan = np.isnan(v)
                if nan and mt == 0:
                    v, nan = 0.0, False
                missing = (mt == 2 and nan) or (mt == 1 and (nan or abs(v) <= ZERO_THRESHOLD))
                go_left = t['dleft'][node] if missing else v <= t['thr'][node]
                child = t['left'][node] if go_left else t['right'][node]
                child_value = t['leaf'][~child] if child < 0 else t['value'][child]
                contrib[f] += child_value - t['value'][node]
                node = child
        return contrib, bias


class NumpyGBMEnsemble:
    """Farklı seed'lerle eğitilmiş modellerin olasılık ortalaması."""

    def __init__(self, members):
        self.members = members
        self.feature_names = members[0].feature_names

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)

    def contributions(self, x):
        """Üyelerin log-odds katkılarının ortalaması."""
        parts = [m.contributions(x) for m in self.members]
        return np.mean([c for c, _ in parts], axis=0), float(np.mean([b for _, b in parts]))

