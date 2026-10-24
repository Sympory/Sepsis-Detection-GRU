"""
Skor ölçeği eşitleme.

Risk sınıfları (config.RISK_LEVELS: Yüksek ≥ 0.35 …) ve klinik kural tabanları (0.25 / 0.35 / 0.60) v4 modelinin
skor ölçeğine göre ayarlıdır. v5'te sepsis örnek ağırlığı 5'ten 1.9'a indiği için ham skorlar yaklaşık yarı
ölçektedir. Model klasöründe score_map.json varsa, ham skor validasyon hastalarındaki SIRASINA göre v4 ölçeğine
çevrilir (yüzdelik eşleme): v5'in %90'lık dilimdeki skoru v4'ün %90'lık dilimdeki skoruna karşılık gelir.

Dönüşüm sırayı korur → ROC-AUC/PR-AUC değişmez; risk sınıfları ve alarm oranları v4 ile tutarlı kalır.
Üretilmesi: scripts/v5/olcek_esitle.py
"""

import json
import os

import numpy as np


class ScoreMap:
    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.y = None if y is None else np.asarray(y, dtype=float)

    @classmethod
    def load(cls, directory):
        path = os.path.join(directory, 'score_map.json')
        if not os.path.exists(path):
            return cls()                       # eşleme yok: ham skor aynen kullanılır (v4)
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return cls(d['x'], d['y'])

    @property
    def active(self):
        return self.x is not None

    def __call__(self, p):
        if not self.active:
            return p
        out = np.interp(np.asarray(p, dtype=float), self.x, self.y)
        return float(out) if np.ndim(p) == 0 else out
