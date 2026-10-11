"""
Sepsis Tahmin Sistemi - Veri Hazırlama Pipeline'ı (DÜZELTİLMİŞ)
================================================================
BU SÜRÜM DATA LEAKAGE SORUNUNU ÇÖZER:
1. GroupShuffleSplit kullanarak hasta bazlı ayrım yapar.
2. Aynı hastanın verisi hem train hem test setine sızmaz.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit # <-- KRİTİK DEĞİŞİKLİK
import argparse
import os
from typing import Tuple


class SepsisDataPreprocessor:
    """Sepsis verilerini önişleme ve sekans oluşturma sınıfı"""
    
    def __init__(self, window_size: int = 6, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size
        self.imputer = None
        self.scaler = None
        self.ohe = None
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        
    def identify_columns(self, df: pd.DataFrame):
        """Veri çerçevesindeki sütun tiplerini belirle"""
        # HEDEF DEĞİŞKENİ VE ŞÜPHELİ SENTETİK ÖZELLİKLERİ HARİÇ TUT
        exclude_cols = [
            'SepsisLabel', 'Patient_ID', 'ICULOS', 'Unnamed: 0',
            # ŞÜPHELİLERİ BURAYA EKLEDİK:
            'PCT', 'CRP', 'Presepsin', 'IL6', 'IL1b', 'ESR', 'MDW', 'MPV', 
            'RDW', 'Neutrophils', 'Lymphocytes', 'DDimer', 'PT', 'aPTT', 
            'INR', 'IonizedCalcium', 'Phosphorus', 'Albumin', 'Sodium', 
            'NLR', 'PLR', 'AnionGap'
        ]
        
        self.categorical_columns = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # 'SepsisLabel' yanlışlıkla kategoriklere girdiyse çıkar
        if 'SepsisLabel' in self.categorical_columns:
            self.categorical_columns.remove('SepsisLabel')

        self.numerical_columns = [
            col for col in df.columns 
            if col not in exclude_cols and col not in self.categorical_columns
        ]
        
        print(f"[OK] {len(self.numerical_columns)} sayısal özellik bulundu")
        print(f"[OK] {len(self.categorical_columns)} kategorik özellik bulundu")
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[1/5] Veri Temizleme...")
        original_len = len(df)
        # SepsisLabel NaN olanları at (Hedef yoksa eğitim olmaz)
        df = df.dropna(subset=['SepsisLabel'])
        
        df = df.drop_duplicates(subset=['Patient_ID', 'ICULOS'])
        print(f"  - {original_len - len(df)} mükerrer/boş kayıt kaldırıldı")
        
        # Fizyolojik temizlik
        if 'HR' in df.columns: df.loc[(df['HR'] < 0) | (df['HR'] > 300), 'HR'] = np.nan
        if 'MAP' in df.columns: df.loc[(df['MAP'] < 0) | (df['MAP'] > 250), 'MAP'] = np.nan
        if 'Temp' in df.columns: df.loc[(df['Temp'] < 25) | (df['Temp'] > 45), 'Temp'] = np.nan
        if 'O2Sat' in df.columns: df.loc[(df['O2Sat'] < 0) | (df['O2Sat'] > 100), 'O2Sat'] = np.nan
            
        df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True)
        return df
    
    def fit_preprocessing(self, df: pd.DataFrame):
        print("\n[2/5] Preprocessing Nesnelerini Eğitiyor...")
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(df[self.numerical_columns])
        print("  [OK] SimpleImputer eğitildi")
        
        X_imputed = self.imputer.transform(df[self.numerical_columns])
        self.scaler = StandardScaler()
        self.scaler.fit(X_imputed)
        print("  [OK] StandardScaler eğitildi")
        
        if self.categorical_columns:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.ohe.fit(df[self.categorical_columns])
            print(f"  [OK] OneHotEncoder eğitildi")
        
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        X_numerical = self.imputer.transform(df[self.numerical_columns])
        X_numerical = self.scaler.transform(X_numerical)
        
        if self.categorical_columns and self.ohe is not None:
            X_categorical = self.ohe.transform(df[self.categorical_columns])
            X_combined = np.hstack([X_numerical, X_categorical])
        else:
            X_combined = X_numerical
            
        return X_combined
    
    def create_sequences(self, df: pd.DataFrame, X_transformed: np.ndarray):
        """Hasta bazlı kayan pencereler oluştur - HASTA ID İLE BERABER DÖNER"""
        print("\n[3/5] Sekans Pencereleri Oluşturuluyor...")
        
        sequences = []
        labels = []
        patient_ids_seq = [] # Hangi sekans hangi hastanın?
        
        unique_patients = df['Patient_ID'].unique()
        print(f"  - {len(unique_patients)} hasta işlenecek")
        
        for i, patient_id in enumerate(unique_patients):
            if (i + 1) % 1000 == 0: print(f"    İşlenen: {i+1}/{len(unique_patients)}", end='\r')
            
            patient_mask = df['Patient_ID'] == patient_id
            patient_X = X_transformed[patient_mask]
            patient_y = df.loc[patient_mask, 'SepsisLabel'].values
            
            # Yeterli veri yoksa atla
            if len(patient_X) < self.window_size:
                continue
                
            for start_idx in range(0, len(patient_X) - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                
                sequence = patient_X[start_idx:end_idx]
                label = patient_y[end_idx - 1]
                
                sequences.append(sequence)
                labels.append(label)
                patient_ids_seq.append(patient_id) # Bu sekansın sahibi kim?
        
        return np.array(sequences), np.array(labels), np.array(patient_ids_seq)
    
    def save_preprocessing_objects(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f: pickle.dump(self.imputer, f)
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f: pickle.dump(self.scaler, f)
        if self.ohe:
            with open(os.path.join(output_dir, 'ohe.pkl'), 'wb') as f: pickle.dump(self.ohe, f)
        
        column_info = {'numerical_columns': self.numerical_columns, 'categorical_columns': self.categorical_columns}
        with open(os.path.join(output_dir, 'column_info.pkl'), 'wb') as f: pickle.dump(column_info, f)
        print(f"\n[OK] Nesneler kaydedildi: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.2)
    args = parser.parse_args()
    
    print("="*60)
    print("SEPSIS VERİ HAZIRLAMA (DÜZELTİLMİŞ - LEAK FREE)")
    print("="*60)
    
    df = pd.read_csv(args.input)
    print(f"Yüklenen Veri: {len(df)} satır")
    
    preprocessor = SepsisDataPreprocessor(window_size=args.window, step_size=args.step)
    preprocessor.identify_columns(df)
    df = preprocessor.clean_data(df)
    preprocessor.fit_preprocessing(df)
    X_transformed = preprocessor.transform_features(df)
    
    # ARTIK HASTA ID'LERİ DE DÖNÜYOR
    X_seq, y_seq, patient_ids_seq = preprocessor.create_sequences(df, X_transformed)
    
    print("\n[4/5] HASTA BAZLI Bölümleme (Data Leakage Önlemi)...")
    
    # 1. TEST SETİNİ AYIR (Hasta bazlı)
    # GroupShuffleSplit, aynı hastanın tüm sekanslarını bir arada tutar
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_val_idx, test_idx = next(splitter_test.split(X_seq, y_seq, groups=patient_ids_seq))
    
    X_train_val = X_seq[train_val_idx]
    y_train_val = y_seq[train_val_idx]
    p_train_val = patient_ids_seq[train_val_idx]
    
    X_test = X_seq[test_idx]
    y_test = y_seq[test_idx]
    
    # 2. VALIDATION SETİNİ AYIR (Hasta bazlı)
    val_ratio = args.val_size / (1 - args.test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(splitter_val.split(X_train_val, y_train_val, groups=p_train_val))
    
    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]
    
    print(f"  [OK] Train Set: {X_train.shape[0]} sekans ({y_train.mean()*100:.2f}% pozitif)")
    print(f"  [OK] Val Set:   {X_val.shape[0]} sekans ({y_val.mean()*100:.2f}% pozitif)")
    print(f"  [OK] Test Set:  {X_test.shape[0]} sekans ({y_test.mean()*100:.2f}% pozitif)")
    print("  -> AYNI HASTA ASLA İKİ FARKLI SETTE YOK!")

    # Kaydet
    print("\n[5/5] Sonuçları Kaydediyor...")
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output, 'y_val.npy'), y_val)
    np.save(os.path.join(args.output, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output, 'y_test.npy'), y_test)
    preprocessor.save_preprocessing_objects(args.output)
    
    print(f"\nTAMAMLANDI. Çıktı: {args.output}")

if __name__ == '__main__':
    main()