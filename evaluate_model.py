"""
Model Evaluation Script - GRU v23 Performance Testing
======================================================

Bu script, eğitilmiş GRU modelini test verisi üzerinde değerlendirir
ve tüm performans metriklerini hesaplar.

Kullanım:
    python evaluate_model.py

Çıktı:
    - ROC-AUC, PR-AUC
    - Confusion Matrix
    - Sensitivity, Specificity, Precision, Recall
    - Detaylı performans raporu
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/gru_v23_best.keras'
TEST_DATA_PATH = 'data/processed/X_test.npy'
TEST_LABELS_PATH = 'data/processed/y_test.npy'
OUTPUT_DIR = 'evaluation_results'

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

def load_test_data():
    """Test verisini yükle"""
    print("📂 Test verisi yükleniyor...")
    X_test = np.load(TEST_DATA_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    
    print(f"   ✓ X_test shape: {X_test.shape}")
    print(f"   ✓ y_test shape: {y_test.shape}")
    print(f"   ✓ Sepsis cases: {y_test.sum():.0f} ({y_test.mean()*100:.2f}%)")
    print(f"   ✓ Normal cases: {(1-y_test).sum():.0f} ({(1-y_test.mean())*100:.2f}%)")
    
    return X_test, y_test

def load_model():
    """Eğitilmiş modeli yükle"""
    print(f"\n🤖 Model yükleniyor: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("   ✓ Model başarıyla yüklendi")
    
    # Model özeti
    print("\n📊 Model Mimarisi:")
    model.summary()
    
    return model

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred_proba, threshold=0.1799):
    """
    Tüm performans metriklerini hesapla
    
    Args:
        y_true: Gerçek etiketler
        y_pred_proba: Model tahmin olasılıkları
        threshold: Sınıflandırma eşiği
    
    Returns:
        metrics dict
    """
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Derived metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # NPV
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # Recall
        'specificity': specificity,
        'precision': precision,      # PPV
        'npv': npv,
        'f1_score': f1,
        'confusion_matrix': cm,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics

def print_metrics_report(metrics, threshold):
    """Metrikleri formatlı şekilde yazdır"""
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANS RAPORU - GRU v23")
    print("="*70)
    
    print(f"\n🎯 Sınıflandırma Eşiği: {threshold:.4f} ({threshold*100:.2f}%)")
    
    print("\n📈 Ana Performans Metrikleri:")
    print(f"   ROC-AUC Score:        {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    print(f"   PR-AUC Score:         {metrics['pr_auc']:.4f}  ({metrics['pr_auc']*100:.2f}%)")
    print(f"   Accuracy (Doğruluk):  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1-Score:             {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    
    print("\n🎯 Klinik Metrikler:")
    print(f"   Sensitivity (Recall): {metrics['sensitivity']:.4f}  ({metrics['sensitivity']*100:.2f}%)")
    print(f"   └─ Gerçek sepsis hastalarının %{metrics['sensitivity']*100:.1f}'ini tespit eder")
    
    print(f"\n   Specificity:          {metrics['specificity']:.4f}  ({metrics['specificity']*100:.2f}%)")
    print(f"   └─ Normal hastaların %{metrics['specificity']*100:.1f}'ini doğru tanımlar")
    
    print(f"\n   Precision (PPV):      {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"   └─ Model 'sepsis' dediğinde %{metrics['precision']*100:.1f} doğrudur")
    
    print(f"\n   NPV:                  {metrics['npv']:.4f}  ({metrics['npv']*100:.2f}%)")
    print(f"   └─ Model 'normal' dediğinde %{metrics['npv']*100:.1f} doğrudur")
    
    print("\n📊 Confusion Matrix:")
    print(f"                      Gerçek Durum")
    print(f"                 ┌──────────┬──────────┐")
    print(f"                 │  Sepsis  │  Normal  │")
    print(f"    ┌────────────┼──────────┼──────────┤")
    print(f"    │  Sepsis    │  {metrics['tp']:>6}  │  {metrics['fp']:>6}  │  ← Model 'Sepsis' dedi")
    print(f"M   │  (Pozitif) │  ✅ TP   │   ❌ FP  │")
    print(f"o   ├────────────┼──────────┼──────────┤")
    print(f"d   │  Normal    │  {metrics['fn']:>6}  │  {metrics['tn']:>6}  │  ← Model 'Normal' dedi")
    print(f"e   │  (Negatif) │  ❌ FN   │   ✅ TN  │")
    print(f"l   └────────────┴──────────┴──────────┘")
    
    print("\n💡 Yorumlama:")
    total_sepsis = metrics['tp'] + metrics['fn']
    total_normal = metrics['tn'] + metrics['fp']
    
    print(f"   • 100 sepsis hastasından {metrics['sensitivity']*100:.0f} tanesi tespit edilir")
    print(f"   • {100 - metrics['sensitivity']*100:.0f} sepsis hastası kaçırılır")
    print(f"   • 100 normal hastanın {100 - metrics['specificity']*100:.0f}'ine yanlış alarm verilir")
    
    print("\n" + "="*70)

def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """ROC Curve grafiği çiz"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - GRU v23', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ ROC curve kaydedildi: {output_path}")
    plt.close()

def plot_pr_curve(y_true, y_pred_proba, output_dir):
    """Precision-Recall Curve grafiği çiz"""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (random classifier at class prevalence)
    baseline = y_true.mean()
    plt.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--',
             label=f'Baseline (prevalence = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Precision-Recall Curve - GRU v23', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ PR curve kaydedildi: {output_path}")
    plt.close()

def save_metrics_to_file(metrics, threshold, output_dir):
    """Metrikleri text dosyasına kaydet"""
    output_path = os.path.join(output_dir, 'performance_metrics.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("GRU v23 Model - Performance Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Threshold: {threshold:.4f}\n\n")
        f.write(f"ROC-AUC:      {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC:       {metrics['pr_auc']:.4f}\n")
        f.write(f"Accuracy:     {metrics['accuracy']:.4f}\n")
        f.write(f"Sensitivity:  {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity:  {metrics['specificity']:.4f}\n")
        f.write(f"Precision:    {metrics['precision']:.4f}\n")
        f.write(f"NPV:          {metrics['npv']:.4f}\n")
        f.write(f"F1-Score:     {metrics['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {metrics['tp']}\n")
        f.write(f"TN: {metrics['tn']}\n")
        f.write(f"FP: {metrics['fp']}\n")
        f.write(f"FN: {metrics['fn']}\n")
    
    print(f"   ✓ Metrikler kaydedildi: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ana evaluation fonksiyonu"""
    print("\n" + "="*70)
    print("GRU v23 MODEL EVALUATION - STARTING")
    print("="*70)
    
    # Output directory oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Test verisini yükle
    X_test, y_test = load_test_data()
    
    # 2. Modeli yükle
    model = load_model()
    
    # 3. Tahmin yap
    print("\nTahminler yapiliyor...")
    y_pred_proba = model.predict(X_test, batch_size=1024, verbose=1)
    y_pred_proba = y_pred_proba.flatten()
    
    print(f"   Tahminler tamamlandi: {len(y_pred_proba)} ornek")
    print(f"   Tahmin araligi: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"   Ortalama tahmin: {y_pred_proba.mean():.4f}")
    
    # 4. Metrikleri hesapla
    print("\nMetrikler hesaplaniyor...")
    threshold = 0.1799  # Gerçek threshold
    metrics = calculate_metrics(y_test, y_pred_proba, threshold)
    
    # 5. Sonuçları göster
    print_metrics_report(metrics, threshold)
    
    # 6. Grafikleri çiz
    print("\nGrafikler olusturuluyor...")
    plot_roc_curve(y_test, y_pred_proba, OUTPUT_DIR)
    plot_pr_curve(y_test, y_pred_proba, OUTPUT_DIR)
    
    # 7. Dosyaya kaydet
    print("\nSonuclar kaydediliyor...")
    save_metrics_to_file(metrics, threshold, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("EVALUATION TAMAMLANDI!")
    print("="*70)
    print(f"\nSonuclar: {OUTPUT_DIR}/")
    print(f"   • performance_metrics.txt")
    print(f"   • roc_curve.png")
    print(f"   • pr_curve.png")
    print()

if __name__ == "__main__":
    main()
