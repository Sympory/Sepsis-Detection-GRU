"""
Deney: v4 zaman serisi özellikleriyle çok katmanlı sinir ağı (MLP).

Özellikler hastanın geçmişini zaten özetlediği için (24 saatlik ortalama, eğilim,
son ölçümden geçen süre...) tekrarlayan katman yerine her saati tek vektör olarak
işleyen bir ağ denenir. Seçim validasyonla; test tahminleri yalnızca kaydedilir.

  py -3.10 scripts/experiments/exp_mlp_v4.py
"""

import json
import os
import pickle
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features_v4 import build_v4

OUT = os.path.join(ROOT, 'models', 'experiments')


class V4Scaler:
    """Medyan doldurma + standartlaştırma + kırpma; yalnızca eğitim hastalarıyla fit edilir."""

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


def build_model(n_features):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    l2 = regularizers.l2(1e-4)
    inp = keras.Input(shape=(n_features,))
    x = layers.GaussianNoise(0.1)(inp)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out, name='sepsis_mlp_v4')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='roc_auc'), keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model


def main():
    t0 = time.time()
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)

    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    F = build_v4(df, group_col='Patient_ID')
    y = df['SepsisLabel'].to_numpy().astype(np.float32)
    tr, va, te = np.load(os.path.join(OUT, 'split_index.npy'))
    scaler = V4Scaler().fit(F[tr])
    X = scaler.transform(F)
    del F, df
    print(f'X {X.shape} · {(time.time() - t0) / 60:.1f} dk', flush=True)

    model = build_model(X.shape[1])
    ckpt = os.path.join(OUT, 'mlp_v4.keras')
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=30, batch_size=4096,
                     class_weight={0: 1.0, 1: 5.0}, verbose=2,
                     callbacks=[keras.callbacks.ModelCheckpoint(ckpt, monitor='val_pr_auc', mode='max', save_best_only=True),
                                keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=4, restore_best_weights=True),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)]).history
    p_val = model.predict(X[va], batch_size=16384, verbose=0).ravel()
    p_test = model.predict(X[te], batch_size=16384, verbose=0).ravel()
    np.save(os.path.join(OUT, 'mlp_v4_val.npy'), p_val)
    np.save(os.path.join(OUT, 'mlp_v4_test.npy'), p_test)
    with open(os.path.join(OUT, 'mlp_v4_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    res = {'val_roc_auc': float(roc_auc_score(y[va], p_val)), 'val_pr_auc': float(average_precision_score(y[va], p_val)),
           'epochs': len(hist['loss']), 'n_features': int(X.shape[1]),
           'history': {k: [round(float(v), 4) for v in vals] for k, vals in hist.items()},
           'minutes': (time.time() - t0) / 60}
    with open(os.path.join(OUT, 'mlp_v4.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
