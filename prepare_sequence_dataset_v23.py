"""
Sepsis Tahmin Sistemi - Veri Hazırlama Pipeline'ı (v23)
========================================================

Bu script, ham ICU verilerini GRU modeli için uygun sekans formatına dönüştürür.

Özellikler:
- Eksik değer doldurma (SimpleImputer)
- Özellik ölçeklendirme (StandardScaler)
- Kategorik değişken kodlama (OneHotEncoder)
- Hasta bazlı 6 saatlik kayan pencere oluşturma
- Train/Validation/Test bölümlemesi

Kullanım:
    python prepare_sequence_dataset_v23.py --input data/train.csv --output data/processed/
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import argparse
import os
from typing import Tuple


class SepsisDataPreprocessor:
    """Sepsis verilerini önişleme ve sekans oluşturma sınıfı"""
    
    def __init__(self, window_size: int = 6, step_size: int = 1):
        """
        Args:
            window_size: Geri bakış penceresi (saat cinsinden)
            step_size: Pencere kayma adımı
        """
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
        # Hedef ve kimlik sütunlarını hariç tut
        exclude_cols = ['SepsisLabel', 'Patient_ID', 'ICULOS']
        
        # Kategorik ve sayısal sütunları ayır
        self.categorical_columns = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.numerical_columns = [
            col for col in df.columns 
            if col not in exclude_cols and col not in self.categorical_columns
        ]
        
        print(f"✓ {len(self.numerical_columns)} sayısal özellik bulundu")
        print(f"✓ {len(self.categorical_columns)} kategorik özellik bulundu")
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi temizle ve geçersiz değerleri kaldır"""
        print("\n[1/5] Veri Temizleme...")
        
        # Mükerrer kayıtları kaldır
        original_len = len(df)
        df = df.drop_duplicates(subset=['Patient_ID', 'ICULOS'])
        print(f"  - {original_len - len(df)} mükerrer kayıt kaldırıldı")
        
        # Fizyolojik olarak geçersiz değerleri temizle
        if 'HR' in df.columns:
            df.loc[(df['HR'] < 0) | (df['HR'] > 300), 'HR'] = np.nan
        if 'MAP' in df.columns:
            df.loc[(df['MAP'] < 0) | (df['MAP'] > 250), 'MAP'] = np.nan
        if 'Temp' in df.columns:
            df.loc[(df['Temp'] < 25) | (df['Temp'] > 45), 'Temp'] = np.nan
        if 'O2Sat' in df.columns:
            df.loc[(df['O2Sat'] < 0) | (df['O2Sat'] > 100), 'O2Sat'] = np.nan
            
        # Hasta ID ve zamana göre sırala
        df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True)
        print(f"  - Toplam {len(df)} kayıt işlenecek")
        
        return df
    
    def fit_preprocessing(self, df: pd.DataFrame):
        """Preprocessing nesnelerini eğit"""
        print("\n[2/5] Preprocessing Nesnelerini Eğitiyor...")
        
        # Eksik değer doldurma (medyan stratejisi)
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(df[self.numerical_columns])
        print("  ✓ SimpleImputer eğitildi")
        
        # Ölçeklendirme
        X_imputed = self.imputer.transform(df[self.numerical_columns])
        self.scaler = StandardScaler()
        self.scaler.fit(X_imputed)
        print("  ✓ StandardScaler eğitildi")
        
        # Kategorik kodlama
        if self.categorical_columns:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.ohe.fit(df[self.categorical_columns])
            print(f"  ✓ OneHotEncoder eğitildi ({len(self.ohe.get_feature_names_out())} özellik)")
        
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Özellikleri dönüştür"""
        # Sayısal özellikleri dönüştür
        X_numerical = self.imputer.transform(df[self.numerical_columns])
        X_numerical = self.scaler.transform(X_numerical)
        
        # Kategorik özellikleri dönüştür
        if self.categorical_columns and self.ohe is not None:
            X_categorical = self.ohe.transform(df[self.categorical_columns])
            X_combined = np.hstack([X_numerical, X_categorical])
        else:
            X_combined = X_numerical
            
        return X_combined
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        X_transformed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hasta bazlı kayan pencereler oluştur"""
        print("\n[3/5] Sekans Pencereleri Oluşturuluyor...")
        
        sequences = []
        labels = []
        patient_ids = df['Patient_ID'].unique()
        
        print(f"  - {len(patient_ids)} hasta işlenecek")
        
        for i, patient_id in enumerate(patient_ids):
            if (i + 1) % 1000 == 0:
                print(f"    İşlenen: {i+1}/{len(patient_ids)} hasta", end='\r')
            
            # Hasta verilerini al
            patient_mask = df['Patient_ID'] == patient_id
            patient_indices = np.where(patient_mask)[0]
            patient_X = X_transformed[patient_mask]
            patient_y = df.loc[patient_mask, 'SepsisLabel'].values
            
            # Kayan pencereler oluştur
            for start_idx in range(0, len(patient_X) - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                
                # 6 saatlik pencere
                sequence = patient_X[start_idx:end_idx]
                
                # Etiket: pencere sonundaki değer
                label = patient_y[end_idx - 1]
                
                sequences.append(sequence)
                labels.append(label)
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        print(f"\n  ✓ {len(sequences)} sekans oluşturuldu")
        print(f"  ✓ Sekans şekli: {X_seq.shape}")
        print(f"  ✓ Pozitif örnekler: {y_seq.sum()} ({100*y_seq.mean():.2f}%)")
        
        return X_seq, y_seq
    
    def save_preprocessing_objects(self, output_dir: str):
        """Preprocessing nesnelerini kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)
        
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        if self.ohe is not None:
            with open(os.path.join(output_dir, 'ohe.pkl'), 'wb') as f:
                pickle.dump(self.ohe, f)
        
        # Sütun bilgilerini kaydet
        column_info = {
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns
        }
        with open(os.path.join(output_dir, 'column_info.pkl'), 'wb') as f:
            pickle.dump(column_info, f)
        
        print(f"\n✓ Preprocessing nesneleri kaydedildi: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Sepsis verilerini sekans formatına dönüştür'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Giriş CSV dosyası yolu'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Çıktı dizini'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=6,
        help='Pencere boyutu (varsayılan: 6 saat)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Adım boyutu (varsayılan: 1)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test seti oranı (varsayılan: 0.2)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Validation seti oranı (varsayılan: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SEPSIS VERİ HAZIRLAMA PIPELINE'I v23")
    print("="*60)
    
    # Veriyi yükle
    print(f"\nVeri yükleniyor: {args.input}")
    df = pd.read_csv(args.input)
    print(f"✓ {len(df)} satır, {len(df.columns)} sütun yüklendi")
    
    # Gerekli sütunları kontrol et
    required_cols = ['Patient_ID', 'ICULOS', 'SepsisLabel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik sütunlar: {missing_cols}")
    
    # Preprocessor oluştur
    preprocessor = SepsisDataPreprocessor(
        window_size=args.window,
        step_size=args.step
    )
    
    # Sütunları belirle
    preprocessor.identify_columns(df)
    
    # Veriyi temizle
    df = preprocessor.clean_data(df)
    
    # Preprocessing nesnelerini eğit
    preprocessor.fit_preprocessing(df)
    
    # Özellikleri dönüştür
    X_transformed = preprocessor.transform_features(df)
    print(f"\n  ✓ Dönüştürülmüş özellik şekli: {X_transformed.shape}")
    
    # Sekansları oluştur
    X_seq, y_seq = preprocessor.create_sequences(df, X_transformed)
    
    # Train/Val/Test ayır
    print("\n[4/5] Veri Setlerini Bölümleme...")
    
    # Önce test setini ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq,
        test_size=args.test_size,
        stratify=y_seq,
        random_state=42
    )
    
    # Sonra validation setini ayır
    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"  ✓ Train: {X_train.shape} ({y_train.mean()*100:.2f}% pozitif)")
    print(f"  ✓ Val:   {X_val.shape} ({y_val.mean()*100:.2f}% pozitif)")
    print(f"  ✓ Test:  {X_test.shape} ({y_test.mean()*100:.2f}% pozitif)")
    
    # Sonuçları kaydet
    print("\n[5/5] Sonuçları Kaydediyor...")
    os.makedirs(args.output, exist_ok=True)
    
    np.save(os.path.join(args.output, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output, 'y_val.npy'), y_val)
    np.save(os.path.join(args.output, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output, 'y_test.npy'), y_test)
    
    print(f"  ✓ NumPy dizileri kaydedildi: {args.output}")
    
    # Preprocessing nesnelerini kaydet
    preprocessor.save_preprocessing_objects(args.output)
    
    print("\n" + "="*60)
    print("VERİ HAZIRLAMA TAMAMLANDI!")
    print("="*60)
    print(f"\nÇıktı dizini: {args.output}")
    print(f"Toplam özellik sayısı: {X_seq.shape[2]}")
    print(f"Pencere boyutu: {args.window} saat")
    print(f"Toplam sekans: {len(X_seq):,}")


if __name__ == '__main__':
    main()
