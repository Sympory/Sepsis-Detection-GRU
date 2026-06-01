"""
NPY -> MAT Donusturucu
======================
Python'daki .npy veri dosyalarini MATLAB'in okuyacagi .mat formatina cevirir.

Kullanim:
    python convert_npy_to_mat.py
"""

import numpy as np
import pickle
import os
import sys

# Encoding hatasi onlemi
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from scipy.io import savemat

def main():
    # Kaynak ve hedef dizinler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    repo_root = os.path.dirname(project_root)
    
    source_dir = os.path.join(repo_root, 'data', 'processed')
    target_dir = os.path.join(project_root, 'data')
    os.makedirs(target_dir, exist_ok=True)
    
    print("=" * 60)
    print("NPY -> MAT DONUSTURUCU")
    print("=" * 60)
    print(f"Kaynak: {source_dir}")
    print(f"Hedef:  {target_dir}")
    
    # Veri dosyalarini yukle
    print("\n[1/3] Veriler yukleniyor...")
    X_train = np.load(os.path.join(source_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(source_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(source_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(source_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(source_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(source_dir, 'y_test.npy'))
    
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    
    # Sutun bilgisi
    print("\n[2/3] Sutun bilgisi yukleniyor...")
    col_info_path = os.path.join(source_dir, 'column_info.pkl')
    with open(col_info_path, 'rb') as f:
        col_info = pickle.load(f)
    
    numerical_columns = col_info['numerical_columns']
    print(f"  {len(numerical_columns)} sayisal sutun: {numerical_columns[:5]}...")
    
    # MAT dosyasina kaydet
    print("\n[3/3] MAT dosyasina kaydediliyor...")
    
    mat_data = {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.float32).reshape(-1, 1),
        'X_val': X_val.astype(np.float32),
        'y_val': y_val.astype(np.float32).reshape(-1, 1),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.float32).reshape(-1, 1),
        'column_names': np.array(numerical_columns, dtype=object),
        'window_size': np.array([X_train.shape[1]]),
        'num_features': np.array([X_train.shape[2]]),
    }
    
    output_path = os.path.join(target_dir, 'sepsis_data.mat')
    savemat(output_path, mat_data, do_compression=True)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  [OK] Kaydedildi: {output_path}")
    print(f"  Dosya boyutu: {file_size_mb:.1f} MB")
    
    # Boyut ozeti
    print("\n" + "=" * 60)
    print("MATLAB FORMAT BILGISI")
    print("=" * 60)
    print(f"  X boyutlari: [N, T, F] = [ornekSayisi, zamanAdimi, ozellikSayisi]")
    print(f"  MATLAB'da permute ile [F, T, N] formatina cevrilecek")
    print(f"  Window size: {X_train.shape[1]}")
    print(f"  Num features: {X_train.shape[2]}")
    print("\nTAMAMLANDI!")

if __name__ == '__main__':
    main()
