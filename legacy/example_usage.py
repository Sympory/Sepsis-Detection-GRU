"""
GRU Sepsis Sistemi - Hızlı Başlangıç Scripti
=============================================

Bu script, tüm pipeline'ı sırayla çalıştırır:
1. Veri hazırlama
2. Model eğitimi
3. Test tahmini

Kullanım:
    python example_usage.py
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Komutu çalıştır ve çıktıyı göster"""
    print("\n" + "="*70)
    print(f"▶ {description}")
    print("="*70)
    print(f"Komut: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ HATA: {description} başarısız oldu!")
        sys.exit(1)
    else:
        print(f"\n✅ {description} tamamlandı!")
    
    return result


def main():
    print("="*70)
    print("GRU SEPSIS TAHMİN SİSTEMİ - ÖRNEK KULLANIM")
    print("="*70)
    print("\nBu script 3 adımda çalışır:")
    print("  1. Veri Hazırlama (prepare_sequence_dataset_v23.py)")
    print("  2. Model Eğitimi (train_gru_v23.py)")
    print("  3. Test Tahmini (run_gru_on_csv_v23.py)")
    print("\n" + "="*70)
    
    # Dizinleri kontrol et ve oluştur
    data_dir = "data"
    processed_dir = os.path.join(data_dir, "processed")
    models_dir = "models"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # ============================================================
    # ADIM 1: VERİ HAZIRLAMA
    # ============================================================
    
    # Not: Bu örnek için gerçek bir veri dosyası belirtmeniz gerekir
    input_csv = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(input_csv):
        print(f"\n⚠️  UYARI: Veri dosyası bulunamadı: {input_csv}")
        print(f"Lütfen eğitim verinizi şu konuma yerleştirin: {input_csv}")
        print("\nVeri formatı örneği:")
        print("Patient_ID,ICULOS,HR,MAP,O2Sat,Temp,Resp,...,SepsisLabel")
        print("P001,1,85,75,98,37.2,18,...,0")
        print("P001,2,88,72,97,37.4,19,...,0")
        print("...")
        
        choice = input("\nYine de devam etmek istiyor musunuz? (e/h): ")
        if choice.lower() != 'e':
            print("Çıkılıyor...")
            sys.exit(0)
    
    prepare_cmd = [
        sys.executable,  # python
        "prepare_sequence_dataset_v23.py",
        "--input", input_csv,
        "--output", processed_dir,
        "--window", "6",
        "--step", "1",
        "--test-size", "0.2",
        "--val-size", "0.2"
    ]
    
    run_command(prepare_cmd, "1. VERİ HAZIRLAMA")
    
    # ============================================================
    # ADIM 2: MODEL EĞİTİMİ
    # ============================================================
    
    train_cmd = [
        sys.executable,
        "train_gru_v23.py",
        "--data", processed_dir,
        "--output", models_dir,
        "--epochs", "60",
        "--batch-size", "512",
        "--lr", "0.001",
        "--gru-units", "64",
        "--dense-units", "32",
        "--dropout", "0.3"
    ]
    
    run_command(train_cmd, "2. MODEL EĞİTİMİ")
    
    # ============================================================
    # ADIM 3: TEST TAHMİNİ
    # ============================================================
    
    # Test için aynı dosyayı kullanabiliriz (gerçek uygulamada farklı olmalı)
    test_csv = input_csv  # veya farklı bir test.csv
    output_csv = "predictions_gru_v23.csv"
    model_path = os.path.join(models_dir, "gru_v23_best.keras")
    
    predict_cmd = [
        sys.executable,
        "run_gru_on_csv_v23.py",
        "--input", test_csv,
        "--output", output_csv,
        "--model", model_path,
        "--preprocessing", processed_dir,
        "--threshold", "0.1799",
        "--window", "6"
    ]
    
    run_command(predict_cmd, "3. TEST TAHMİNİ")
    
    # ============================================================
    # ÖZET
    # ============================================================
    
    print("\n" + "="*70)
    print("✅ TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!")
    print("="*70)
    print("\nOluşturulan Dosyalar:")
    print(f"  📁 İşlenmiş Veri: {processed_dir}/")
    print(f"  🧠 Eğitilmiş Model: {model_path}")
    print(f"  📊 Tahminler: {output_csv}")
    print(f"  📈 Eğitim Grafiği: {models_dir}/training_history.png")
    print(f"  📋 Test Sonuçları: {models_dir}/test_results.json")
    
    print("\nSonraki Adımlar:")
    print("  1. Eğitim grafiklerini inceleyin: training_history.png")
    print("  2. Test metriklerini kontrol edin: test_results.json")
    print("  3. Tahminleri analiz edin: predictions_gru_v23.csv")
    print("  4. Yeni hastalara tahmin yapmak için run_gru_on_csv_v23.py kullanın")
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
