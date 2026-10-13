"""
Sepsis Tahmin Sistemi - GRU Model Eğitimi (v23)
================================================

Bu script, önceden hazırlanmış sekans verilerini kullanarak GRU modelini eğitir.

Model Mimarisi:
- GRU(64) → BatchNorm → Dropout(0.3)
- Dense(32, ReLU) → Dropout(0.3)
- Dense(1, Sigmoid)

Özellikler:
- Class weighting (dengesiz veri için)
- EarlyStopping (val_pr_auc)
- ReduceLROnPlateau
- ModelCheckpoint (en iyi ağırlıkları kaydet)

Kullanım:
    python train_gru_v23.py --data data/processed/ --epochs 60
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime


class GRUSepsisModel:
    """GRU tabanlı sepsis tahmin modeli"""
    
    def __init__(self, input_shape=(6, 63), gru_units=64, dense_units=32, dropout_rate=0.3):
        """
        Args:
            input_shape: (sequence_length, num_features)
            gru_units: GRU katmanındaki ünite sayısı
            dense_units: Dense katmanındaki ünite sayısı
            dropout_rate: Dropout oranı
        """
        self.input_shape = input_shape
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """GRU modelini oluştur"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Recurrent encoder
            layers.GRU(
                self.gru_units,
                return_sequences=False,
                name='gru_layer'
            ),
            layers.BatchNormalization(name='batch_norm'),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # Dense classifier
            layers.Dense(
                self.dense_units,
                activation='relu',
                name='dense_hidden'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='GRU_Sepsis_v23')
        
        self.model = model
        print("\n" + "="*60)
        print("MODEL MİMARİSİ")
        print("="*60)
        self.model.summary()
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Modeli derle"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                keras.metrics.AUC(name='roc_auc', curve='ROC'),
                keras.metrics.AUC(name='pr_auc', curve='PR'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        print(f"\n✓ Model derlendi (lr={learning_rate})")
    
    def calculate_class_weights(self, y_train):
        """Sınıf ağırlıklarını hesapla"""
        n_samples = len(y_train)
        n_positive = np.sum(y_train)
        n_negative = n_samples - n_positive
        
        weight_negative = n_samples / (2 * n_negative)
        weight_positive = n_samples / (2 * n_positive)
        
        class_weight = {0: weight_negative, 1: weight_positive}
        
        print(f"\nSınıf Ağırlıkları:")
        print(f"  Negatif (0): {weight_negative:.4f}")
        print(f"  Pozitif (1): {weight_positive:.4f}")
        print(f"  Ağırlık oranı: {weight_positive/weight_negative:.2f}x")
        
        return class_weight
    
    def train(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=60,
        batch_size=512,
        class_weight=None,
        callbacks_list=None
    ):
        """Modeli eğit"""
        print("\n" + "="*60)
        print("MODEL EĞİTİMİ BAŞLIYOR")
        print("="*60)
        print(f"Train örnekleri: {len(X_train):,}")
        print(f"Validation örnekleri: {len(X_val):,}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Test setini değerlendir"""
        print("\n" + "="*60)
        print("TEST SETİ DEĞERLENDİRMESİ")
        print("="*60)
        
        # Model tahminleri
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrikler
        test_loss, test_roc_auc, test_pr_auc, test_precision, test_recall = \
            self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Metrikleri:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  ROC-AUC:   {test_roc_auc:.4f}")
        print(f"  PR-AUC:    {test_pr_auc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
        
        # Classification report
        print(f"\nDetaylı Rapor:")
        print(classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif']))
        
        results = {
            'test_loss': float(test_loss),
            'test_roc_auc': float(test_roc_auc),
            'test_pr_auc': float(test_pr_auc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'confusion_matrix': cm.tolist()
        }
        
        return results, y_pred_proba
    
    def plot_training_history(self, save_path=None):
        """Eğitim geçmişini görselleştir"""
        history_dict = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRU Model Eğitim Geçmişi', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history_dict['loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history_dict['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC-AUC
        axes[0, 1].plot(history_dict['roc_auc'], label='Train', linewidth=2)
        axes[0, 1].plot(history_dict['val_roc_auc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('ROC-AUC Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PR-AUC
        axes[1, 0].plot(history_dict['pr_auc'], label='Train', linewidth=2)
        axes[1, 0].plot(history_dict['val_pr_auc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('PR-AUC Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PR-AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 1].plot(history_dict['precision'], label='Train Precision', linewidth=2)
        axes[1, 1].plot(history_dict['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
        axes[1, 1].plot(history_dict['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history_dict['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Grafik kaydedildi: {save_path}")
        
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='GRU modelini eğit'
    )
    # 'required=True' kısmını kaldırdık ve dosya yolunu 'default' olarak verdik
    parser.add_argument(
        '--data',
        type=str,
        default='data/train.csv',
        help='Veri yolu'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Model çıktı dizini'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60,
        help='Maksimum epoch sayısı'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch boyutu'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Öğrenme oranı'
    )
    parser.add_argument(
        '--gru-units',
        type=int,
        default=64,
        help='GRU ünite sayısı'
    )
    parser.add_argument(
        '--dense-units',
        type=int,
        default=32,
        help='Dense katman ünite sayısı'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout oranı'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GRU SEPSIS TAHMİN MODELİ - EĞİTİM v23")
    print("="*60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Veriyi yükle
    print(f"\nVeri yükleniyor: {args.data}")
    X_train = np.load(os.path.join(args.data, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data, 'y_train.npy'))
    X_val = np.load(os.path.join(args.data, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data, 'y_val.npy'))
    X_test = np.load(os.path.join(args.data, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data, 'y_test.npy'))
    
    print(f"✓ Veri yüklendi")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Model oluştur
    input_shape = (X_train.shape[1], X_train.shape[2])
    gru_model = GRUSepsisModel(
        input_shape=input_shape,
        gru_units=args.gru_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout
    )
    
    gru_model.build_model()
    gru_model.compile_model(learning_rate=args.lr)
    
    # Sınıf ağırlıklarını hesapla
    class_weight = gru_model.calculate_class_weights(y_train)
    
    # Callbacks tanımla
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, 'gru_v23_best.keras')
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_pr_auc',
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=4,
            mode='max',
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_pr_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Modeli eğit
    history = gru_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks_list=callbacks_list
    )
    
    # Eğitim geçmişini kaydet
    history_path = os.path.join(args.output, 'training_history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"\n✓ Eğitim geçmişi kaydedildi: {history_path}")
    
    # Grafikleri çiz
    plot_path = os.path.join(args.output, 'training_history.png')
    gru_model.plot_training_history(save_path=plot_path)
    
    # Test setini değerlendir
    test_results, y_pred_proba = gru_model.evaluate(X_test, y_test)
    
    # Test sonuçlarını kaydet
    results_path = os.path.join(args.output, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ Test sonuçları kaydedildi: {results_path}")
    
    # Tahminleri kaydet
    predictions_path = os.path.join(args.output, 'test_predictions.npy')
    np.save(predictions_path, y_pred_proba)
    print(f"✓ Tahminler kaydedildi: {predictions_path}")
    
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI!")
    print("="*60)
    print(f"Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"En iyi model: {model_path}")
    print(f"Test ROC-AUC: {test_results['test_roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_results['test_pr_auc']:.4f}")


if __name__ == '__main__':
    main()
