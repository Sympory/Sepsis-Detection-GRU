"""
Deney: daha uzun geriye bakış penceresi (ve isteğe bağlı v4 özellikleri) ile GRU.

Pencereler belleğe topluca yazılmaz; her batch'te hasta içindeki satır
indekslerinden anında üretilir (24 saat × 1,5M satır belleğe sığmadığı için).
Seçim validasyonla yapılır; test tahminleri yalnızca kaydedilir.

  py -3.10 scripts/experiments/exp_gru_window.py --window 24
  py -3.10 scripts/experiments/exp_gru_window.py --window 12 --features v4top
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = os.path.join(ROOT, 'models', 'experiments')


def window_index(pid, window):
    """Her satır için penceresindeki satır indeksleri (hasta başlangıcında ilk satır tekrarlanır)."""
    n = len(pid)
    change = np.r_[True, pid[1:] != pid[:-1]]
    first = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    offs = np.arange(-window + 1, 1)
    idx = np.arange(n)[:, None] + offs[None, :]
    return np.maximum(idx, first[:, None])


def make_dataset(X, idx_rows, y, batch, shuffle):
    import tensorflow as tf
    rows = np.asarray(idx_rows)

    def gen():
        order = np.random.permutation(len(rows)) if shuffle else np.arange(len(rows))
        for s in range(0, len(order), batch):
            b = rows[order[s:s + batch]]
            yield X[b], y[b[:, -1]]

    sig = (tf.TensorSpec((None, rows.shape[1], X.shape[1]), tf.float32), tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(4)


def build_model(window, n_features, units=64, dropout=0.4):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    l2 = regularizers.l2(1e-4)
    inp = keras.Input(shape=(window, n_features))
    x = layers.GaussianNoise(0.1)(inp)
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True, kernel_regularizer=l2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(units, kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(5e-4, clipnorm=1.0), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='roc_auc'), keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--features', choices=['v3', 'v4top'], default='v3')
    ap.add_argument('--top', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    args = ap.parse_args()
    tag = f'gru_w{args.window}_{args.features}'
    t0 = time.time()

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)
    from features import FeatureTransformer, build_hourly_features
    from train_v3 import split_patients

    df = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True).dropna(subset=['SepsisLabel'])
    parts = split_patients(df)
    pid = df['Patient_ID'].to_numpy()
    tr = np.isin(pid, list(parts['train']))
    va = np.isin(pid, list(parts['val']))
    te = np.isin(pid, list(parts['test']))
    y = df['SepsisLabel'].to_numpy().astype(np.float32)

    if args.features == 'v3':
        feats = build_hourly_features(df, group_col='Patient_ID')
        X = FeatureTransformer().fit(feats[tr]).transform(feats)
    else:
        from features_v4 import build_v4
        # Önem sırası ayrı süreçte yazılır: TensorFlow yüklüyken LightGBM'in bazı C fonksiyonları çöküyor
        order = json.load(open(os.path.join(OUT, 'lgbm_v4_importance.json'), encoding='utf-8'))
        feats = build_v4(df, group_col='Patient_ID')
        cols = order[:args.top]
        cols += [c for c in ('Age', 'Gender', 'Hour') if c not in cols]
        F = feats[cols]
        med = F[tr].median()
        F = F.fillna(med).fillna(0)
        mu, sd = F[tr].mean(), F[tr].std().replace(0, 1)
        X = ((F - mu) / sd).clip(-6, 6).to_numpy(np.float32)
    del df
    print(f'{tag}: X {X.shape} · {(time.time() - t0) / 60:.1f} dk', flush=True)

    W = window_index(pid, args.window)
    ds_tr = make_dataset(X, W[tr], y, args.batch, True)
    ds_va = make_dataset(X, W[va], y, 4096, False)
    model = build_model(args.window, X.shape[1])
    ckpt = os.path.join(OUT, f'{tag}.keras')
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, class_weight={0: 1.0, 1: 5.0}, verbose=2,
                     callbacks=[keras.callbacks.ModelCheckpoint(ckpt, monitor='val_pr_auc', mode='max', save_best_only=True),
                                keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=3, restore_best_weights=True),
                                keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=2)]).history
    p_val = model.predict(make_dataset(X, W[va], y, 4096, False), verbose=0).ravel()
    p_test = model.predict(make_dataset(X, W[te], y, 4096, False), verbose=0).ravel()
    np.save(os.path.join(OUT, f'{tag}_val.npy'), p_val)
    np.save(os.path.join(OUT, f'{tag}_test.npy'), p_test)
    res = {'val_roc_auc': float(roc_auc_score(y[va], p_val)), 'val_pr_auc': float(average_precision_score(y[va], p_val)),
           'epochs': len(hist['loss']), 'n_features': int(X.shape[1]), 'window': args.window,
           'history': {k: [round(float(v), 4) for v in vals] for k, vals in hist.items()},
           'minutes': (time.time() - t0) / 60}
    with open(os.path.join(OUT, f'{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
