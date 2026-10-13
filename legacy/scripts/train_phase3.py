"""
Phase 3 Wrapper: Focal Loss Model Eğitimi (v25) - DÜZELTİLMİŞ YOL
=================================================================
Bu script, train_gru_v25_focal.py dosyasını çalıştırır.
Veri yolu: data/processed_56/ (Senin hazırladığın yer)
"""

import os
import sys
from unittest.mock import patch

# 1. YOL AYARLAMALARI
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# 2. IMPORT KONTROLÜ
try:
    from train_gru_v25_focal import main as experimental_main
except ImportError as e:
    print("\nKRİTİK HATA: 'train_gru_v25_focal.py' dosyası bulunamadı.")
    sys.exit(1)

def main():
    print("="*60)
    print("PHASE 4: FOCAL LOSS MODEL EĞİTİMİ (LEAK-FREE VERİ İLE)")
    print("="*60)
    
    # 3. ARGÜMANLAR (DÜZELTİLDİ: data/processed_56/)
    args = [
        'train_gru_v25_focal.py',
        '--data', 'data/processed_56/',        # <-- SENİN VERİN BURADA
        '--output', 'models/gru_v25_focal/',   # Yeni model buraya kaydedilecek
        '--epochs', '60',
        '--batch-size', '512',
        '--lr', '0.001'
    ]
    
    print(f"Model Kayıt Yeri: models/gru_v25_focal/")
    print(f"Veri Kaynağı: data/processed_56/")
    
    with patch.object(sys, 'argv', args):
        experimental_main()

if __name__ == '__main__':
    main()