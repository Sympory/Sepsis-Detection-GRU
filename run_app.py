# -*- coding: utf-8 -*-
"""
GRU-based Sepsis Early Detection System - Flask API
This wrapper runs app.py with UTF-8 encoding for console output
"""

import sys
import io
# -*- coding: utf-8 -*-
"""
GRU-based Sepsis Early Detection System - Flask API
This wrapper runs app.py with UTF-8 encoding for console output
"""

import sys
import io
import traceback

# Force UTF-8 encoding for stdout/stderr to prevent unicode errors on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from app import app, init_database, load_model_and_preprocessing

    if __name__ == '__main__':
        print("Veritabanı başlatılıyor...")
        init_database()
        
        print("Model yükleniyor...")
        if load_model_and_preprocessing():
            print("Sunucu başlatılıyor...")
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("Model yüklenemediği için sunucu başlatılamadı.")
            input("Çıkmak için Enter'a basın...")

except Exception as e:
    print("\nKRİTİK HATA OLUŞTU:")
    print("-" * 60)
    traceback.print_exc()
    print("-" * 60)
    input("Çıkmak için Enter'a basın...")
