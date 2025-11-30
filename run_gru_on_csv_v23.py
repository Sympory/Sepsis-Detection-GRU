"""
Sepsis Tahmin Sistemi - İnference Pipeline (v23)
=================================================

Bu script, eğitilmiş GRU modelini kullanarak yeni hasta verilerine tahmin yapar.

Özellikler:
- Eğitilmiş model ve preprocessing nesnelerini yükler
- CSV formatındaki hasta verilerini işler
- Hasta bazlı 6 saatlik kayan pencereler oluşturur
- Her saat için sepsis risk skorunu hesaplar
- Sonuçları CSV olarak kaydeder

Kullanım:
    python run_gru_on_csv_v23.py --input test_data.csv --model models/gru_v23_best.keras --preprocessing data/processed/
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import argparse
import os
from datetime import datetime
from typing import List, Tuple


class SepsisInferencePipeline:
    """GRU modeli için inference pipeline'ı"""
    
    def __init__(
        self,
        model_path: str,
        preprocessing_dir: str,
        window_size: int = 6,
        threshold: float = 0.1799
    ):
        """
        Args:
            model_path: Eğitilmiş model dosyası yolu
            preprocessing_dir: Preprocessing nesnelerinin bulunduğu dizin
            window_size: Sekans pencere boyutu
            threshold: Sınıflandırma eşiği
        """
        self.model_path = model_path
        self.preprocessing_dir = preprocessing_dir
        self.window_size = window_size
        self.threshold = threshold
        
        self.model = None
        self.imputer = None
        self.scaler = None
        self.ohe = None
        self.numerical_columns = None
        self.categorical_columns = None
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        print(f"\nModel yükleniyor: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("✓ Model yüklendi")
        
    def load_preprocessing_objects(self):
        """Preprocessing nesnelerini yükle"""
        print(f"\nPreprocessing nesneleri yükleniyor: {self.preprocessing_dir}")
        
        # Imputer
        with open(os.path.join(self.preprocessing_dir, 'imputer.pkl'), 'rb') as f:
            self.imputer = pickle.load(f)
        
        # Scaler
        with open(os.path.join(self.preprocessing_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # OneHotEncoder (opsiyonel)
        ohe_path = os.path.join(self.preprocessing_dir, 'ohe.pkl')
        if os.path.exists(ohe_path):
            with open(ohe_path, 'rb') as f:
                self.ohe = pickle.load(f)
        
        # Sütun bilgileri
        with open(os.path.join(self.preprocessing_dir, 'column_info.pkl'), 'rb') as f:
            column_info = pickle.load(f)
            self.numerical_columns = column_info['numerical_columns']
            self.categorical_columns = column_info['categorical_columns']
        
        print("✓ Preprocessing nesneleri yüklendi")
        print(f"  - Sayısal özellikler: {len(self.numerical_columns)}")
        print(f"  - Kategorik özellikler: {len(self.categorical_columns)}")
        
    def preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame'i model girdisine dönüştür"""
        # Sayısal özellikleri işle
        X_numerical = df[self.numerical_columns].values
        X_numerical = self.imputer.transform(X_numerical)
        X_numerical = self.scaler.transform(X_numerical)
        
        # Kategorik özellikleri işle
        if self.categorical_columns and self.ohe is not None:
            X_categorical = self.ohe.transform(df[self.categorical_columns])
            X_combined = np.hstack([X_numerical, X_categorical])
        else:
            X_combined = X_numerical
        
        return X_combined
    
    def create_patient_sequences(
        self,
        patient_data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Bir hasta için kayan pencere sekansları oluştur"""
        # Preprocessing uygula
        X_processed = self.preprocess_dataframe(patient_data)
        
        sequences = []
        row_indices = []
        
        # Kayan pencereler oluştur
        for i in range(len(X_processed)):
            if i < self.window_size - 1:
                # Yeterli geçmiş yok - None olarak işaretle veya atla
                sequences.append(None)
                row_indices.append(i)
            else:
                # 6 saatlik pencere
                start_idx = i - self.window_size + 1
                end_idx = i + 1
                sequence = X_processed[start_idx:end_idx]
                
                sequences.append(sequence)
                row_indices.append(i)
        
        return sequences, row_indices
    
    def predict_patient(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """Bir hasta için tahmin yap"""
        sequences, row_indices = self.create_patient_sequences(patient_data)
        
        predictions = []
        
        for i, seq in enumerate(sequences):
            if seq is None:
                # Yeterli geçmiş yok
                predictions.append({
                    'row_index': row_indices[i],
                    'proba': None,
                    'yhat': None,
                    'insufficient_history': True
                })
            else:
                # Model tahmini
                X_seq = np.expand_dims(seq, axis=0)  # (1, 6, features)
                risk_score = self.model.predict(X_seq, verbose=0)[0, 0]
                prediction = 1 if risk_score >= self.threshold else 0
                
                predictions.append({
                    'row_index': row_indices[i],
                    'proba': float(risk_score),
                    'yhat': int(prediction),
                    'insufficient_history': False
                })
        
        return pd.DataFrame(predictions)
    
    def predict_csv(self, input_path: str, output_path: str):
        """CSV dosyasındaki tüm hastalar için tahmin yap"""
        print("\n" + "="*60)
        print("TAHMİN PIPELINE'I BAŞLATILIYOR")
        print("="*60)
        print(f"Giriş dosyası: {input_path}")
        print(f"Çıkış dosyası: {output_path}")
        
        # Veriyi yükle
        print(f"\nVeri yükleniyor...")
        df = pd.read_csv(input_path)
        print(f"✓ {len(df)} satır yüklendi")
        
        # Hasta ID'lerini belirle
        if 'Patient_ID' not in df.columns:
            raise ValueError("'Patient_ID' sütunu bulunamadı!")
        
        patient_ids = df['Patient_ID'].unique()
        print(f"✓ {len(patient_ids)} benzersiz hasta bulundu")
        
        # Her hasta için tahmin yap
        print(f"\nHastalar işleniyor...")
        all_predictions = []
        
        for i, patient_id in enumerate(patient_ids):
            if (i + 1) % 100 == 0:
                print(f"  İşlenen: {i+1}/{len(patient_ids)} hasta", end='\r')
            
            # Hasta verilerini al
            patient_mask = df['Patient_ID'] == patient_id
            patient_df = df[patient_mask].copy()
            
            # ICULOS'a göre sırala (varsa)
            if 'ICULOS' in patient_df.columns:
                patient_df = patient_df.sort_values('ICULOS')
            
            # Tahminleri yap
            patient_predictions = self.predict_patient(patient_df)
            
            # Hasta ID'sini ekle
            patient_predictions['Patient_ID'] = patient_id
            
            # Orijinal satırlarla birleştir için gerekli bilgileri ekle
            if 'ICULOS' in patient_df.columns:
                patient_predictions['ICULOS'] = patient_df['ICULOS'].values
            
            all_predictions.append(patient_predictions)
        
        print(f"\n✓ Tüm hastalar işlendi")
        
        # Tüm tahminleri birleştir
        results_df = pd.concat(all_predictions, ignore_index=True)
        
        # Sonuçları kaydet
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Tahminler kaydedildi: {output_path}")
        
        # İstatistikler
        valid_predictions = results_df[~results_df['insufficient_history']]
        if len(valid_predictions) > 0:
            print(f"\nTahmin İstatistikleri:")
            print(f"  Toplam satır: {len(results_df):,}")
            print(f"  Geçerli tahmin: {len(valid_predictions):,}")
            print(f"  Yetersiz geçmiş: {results_df['insufficient_history'].sum():,}")
            print(f"  Ortalama risk skoru: {valid_predictions['proba'].mean():.4f}")
            print(f"  Pozitif tahminler: {valid_predictions['yhat'].sum():,} ({100*valid_predictions['yhat'].mean():.2f}%)")
            print(f"  Risk skoru dağılımı:")
            print(f"    Min:  {valid_predictions['proba'].min():.4f}")
            print(f"    25%:  {valid_predictions['proba'].quantile(0.25):.4f}")
            print(f"    50%:  {valid_predictions['proba'].quantile(0.50):.4f}")
            print(f"    75%:  {valid_predictions['proba'].quantile(0.75):.4f}")
            print(f"    Max:  {valid_predictions['proba'].max():.4f}")
        
        return results_df
    
    def run(self, input_path: str, output_path: str):
        """Tam pipeline'ı çalıştır"""
        start_time = datetime.now()
        print("="*60)
        print("GRU SEPSIS TAHMİN SİSTEMİ - İNFERENCE v23")
        print("="*60)
        print(f"Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model ve preprocessing nesnelerini yükle
        self.load_model()
        self.load_preprocessing_objects()
        
        # Tahminleri yap
        results = self.predict_csv(input_path, output_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("TAHMİN TAMAMLANDI!")
        print("="*60)
        print(f"Bitiş: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Süre: {duration:.2f} saniye")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Eğitilmiş GRU modeli ile sepsis tahmini yap'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Giriş CSV dosyası'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions_gru_v23.csv',
        help='Çıkış CSV dosyası'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Eğitilmiş model (.keras dosyası)'
    )
    parser.add_argument(
        '--preprocessing',
        type=str,
        required=True,
        help='Preprocessing nesnelerinin dizini'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1799,
        help='Sınıflandırma eşiği (varsayılan: 0.1799)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=6,
        help='Sekans pencere boyutu (varsayılan: 6)'
    )
    
    args = parser.parse_args()
    
    # Pipeline oluştur
    pipeline = SepsisInferencePipeline(
        model_path=args.model,
        preprocessing_dir=args.preprocessing,
        window_size=args.window,
        threshold=args.threshold
    )
    
    # Çalıştır
    pipeline.run(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
