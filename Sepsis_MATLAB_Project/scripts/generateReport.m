function generateReport(results, history, outputDir)
% GENERATEREPORT Otomatik proje raporu oluşturur (TXT formatında)
%   generateReport(results, history, outputDir)
%
%   MATLAB Command Window'da ve dosyaya kapsamlı bir rapor yazar.
%   TÜBİTAK sunumu ve ders projesi için kullanılabilir.

    if nargin < 3
        outputDir = 'results';
    end
    
    reportPath = fullfile(outputDir, 'proje_raporu.txt');
    fid = fopen(reportPath, 'w', 'n', 'UTF-8');
    
    % Hem dosyaya hem ekrana yazan fonksiyon
    printBoth = @(fmt, varargin) dualPrint(fid, fmt, varargin{:});
    
    printBoth('╔══════════════════════════════════════════════════════════════╗\n');
    printBoth('║     SEPSİS ERKEN TAHMİN SİSTEMİ — PROJE RAPORU            ║\n');
    printBoth('║     MATLAB Deep Learning Toolbox ile Geliştirilmiştir       ║\n');
    printBoth('╚══════════════════════════════════════════════════════════════╝\n');
    printBoth('\nTarih: %s\n', datestr(now, 'dd.mm.yyyy HH:MM'));
    
    % --- 1. PROJE TANIMI ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('1. PROJE TANIMI\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('Amaç: Yoğun bakım ünitesinde (YBÜ) yatan hastaların vital\n');
    printBoth('bulgularını analiz ederek Sepsis''i erken aşamada tahmin etmek.\n');
    printBoth('\nVeri Seti: PhysioNet/CinC Challenge 2019 (Sepsis Prediction)\n');
    printBoth('  - 40.000+ hasta, 40 klinik özellik, saatlik ölçümler\n');
    printBoth('  - Dengesiz veri: ~%%2 pozitif (Sepsis) / %%98 negatif\n');
    
    % --- 2. MODEL MİMARİSİ ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('2. MODEL MİMARİSİ (Enhanced v2)\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('  ┌─────────────────────────────────────────────┐\n');
    printBoth('  │  Girdi: [52 özellik × 6 saatlik pencere]     │\n');
    printBoth('  │         (40 orijinal + 12 mühendislik)        │\n');
    printBoth('  ├─────────────────────────────────────────────┤\n');
    printBoth('  │  Bi-LSTM Katman 1 (96 ünite, çift yönlü)     │\n');
    printBoth('  │  → Layer Normalization → Dropout(0.3)         │\n');
    printBoth('  ├─────────────────────────────────────────────┤\n');
    printBoth('  │  Bi-LSTM Katman 2 (64 ünite, çift yönlü)     │\n');
    printBoth('  │  → Layer Normalization → Dropout(0.3)         │\n');
    printBoth('  ├─────────────────────────────────────────────┤\n');
    printBoth('  │  Multi-Head Temporal Attention                │\n');
    printBoth('  │  (128 ünite, 4 dikkat başı)                   │\n');
    printBoth('  ├─────────────────────────────────────────────┤\n');
    printBoth('  │  Dense(64, GELU) → BN → Dropout(0.3)         │\n');
    printBoth('  │  Dense(32, GELU) → BN → Dropout(0.2)         │\n');
    printBoth('  │  Dense(1, Sigmoid) → Çıktı                   │\n');
    printBoth('  └─────────────────────────────────────────────┘\n');
    
    % --- 3. FEATURE ENGINEERING ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('3. FEATURE ENGINEERING (12 Yeni Özellik)\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('  Delta Özellikleri (Değişim Hızı):\n');
    printBoth('    dHR, dTemp, dMAP, dLactate, dResp, dO2Sat\n');
    printBoth('  Etkileşim Özellikleri:\n');
    printBoth('    ShockIndex (HR/SBP), MAP/HR, O2 Delivery, Resp/HR\n');
    printBoth('  Değişkenlik Özellikleri:\n');
    printBoth('    HR Variability, Temperature Variability\n');
    
    % --- 4. EĞİTİM DETAYLARI ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('4. EĞİTİM DETAYLARI\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('  Kayıp Fonksiyonu:  Focal Loss (γ=2.0, α=0.25)\n');
    printBoth('  Optimizer:         Adam\n');
    printBoth('  Öğrenme Oranı:     0.001 (ReduceLROnPlateau ile)\n');
    printBoth('  Early Stopping:    patience=10 (PR-AUC izleme)\n');
    if isfield(history, 'bestEpoch')
        printBoth('  Eğitim Süresi:     %.1f dakika\n', history.trainingTime/60);
        printBoth('  Toplam Epoch:      %d (en iyi: %d)\n', ...
            length(history.trainLoss), history.bestEpoch);
    end
    
    % --- 5. TEST SONUÇLARI ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('5. TEST SETİ SONUÇLARI\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('\n  %-20s │ %-10s\n', 'Metrik', 'Değer');
    printBoth('  %s\n', repmat('─', 1, 35));
    printBoth('  %-20s │ %.4f\n', 'ROC-AUC', results.rocAUC);
    printBoth('  %-20s │ %.4f\n', 'PR-AUC', results.prAUC);
    printBoth('  %-20s │ %.4f\n', 'F1-Score', results.f1Score);
    printBoth('  %-20s │ %.4f\n', 'Accuracy', results.accuracy);
    printBoth('  %-20s │ %.4f\n', 'Sensitivity', results.sensitivity);
    printBoth('  %-20s │ %.4f\n', 'Specificity', results.specificity);
    printBoth('  %-20s │ %.4f\n', 'Precision', results.precision);
    
    printBoth('\n  Confusion Matrix:\n');
    printBoth('  ┌──────────┬──────────┐\n');
    printBoth('  │ TN: %-5d│ FP: %-5d│\n', results.TN, results.FP);
    printBoth('  ├──────────┼──────────┤\n');
    printBoth('  │ FN: %-5d│ TP: %-5d│\n', results.FN, results.TP);
    printBoth('  └──────────┴──────────┘\n');
    
    % --- 6. YÖNTEM ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('6. KULLANILAN YÖNTEMLER\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('  • Bidirectional LSTM (Bi-LSTM): İleri ve geri yönde\n');
    printBoth('    temporal bağımlılıkları yakalamak için\n');
    printBoth('  • Multi-Head Attention: Hangi zaman adımlarının\n');
    printBoth('    Sepsis için kritik olduğunu öğrenmek için\n');
    printBoth('  • Focal Loss: Dengesiz sınıf dağılımını kompanse\n');
    printBoth('    etmek ve zor örneklere odaklanmak için\n');
    printBoth('  • Feature Engineering: Delta, etkileşim ve\n');
    printBoth('    değişkenlik özellikleri ile tahmin gücünü artırmak\n');
    
    % --- 7. ARAÇLAR ---
    printBoth('\n%s\n', repmat('─', 1, 60));
    printBoth('7. KULLANILAN ARAÇLAR\n');
    printBoth('%s\n', repmat('─', 1, 60));
    printBoth('  • MATLAB R2025b — Deep Learning Toolbox\n');
    printBoth('  • dlnetwork — Custom training loop\n');
    printBoth('  • Adam Optimizer ile otomatik diferansiyasyon\n');
    printBoth('  • perfcurve — ROC ve PR eğrisi analizi\n');
    
    printBoth('\n%s\n', repmat('═', 1, 60));
    printBoth('RAPOR SONU\n');
    printBoth('%s\n', repmat('═', 1, 60));
    
    fclose(fid);
    fprintf('\n  [OK] Rapor kaydedildi: %s\n', reportPath);
end


function dualPrint(fid, fmt, varargin)
% Hem konsola hem dosyaya yazdırır
    str = sprintf(fmt, varargin{:});
    fprintf('%s', str);
    fprintf(fid, '%s', str);
end
