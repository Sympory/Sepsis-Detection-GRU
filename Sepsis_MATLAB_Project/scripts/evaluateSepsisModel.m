function results = evaluateSepsisModel(net, data, outputDir)
% EVALUATESEPSISMODEL Test seti üzerinde kapsamlı model değerlendirmesi
%   results = evaluateSepsisModel(net, data, outputDir)
%
%   Hesaplanan Metrikler:
%     - ROC-AUC, PR-AUC
%     - Confusion Matrix (TN, FP, FN, TP)
%     - Precision, Recall (Sensitivity), Specificity
%     - F1-Score (optimal threshold ile)
%     - Accuracy
%
%   Girdi:
%     net       - Eğitilmiş dlnetwork
%     data      - struct (loadSepsisData çıktısı)
%     outputDir - Sonuçların kaydedileceği klasör
%
%   Çıktı:
%     results   - Tüm metrikleri içeren struct

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('TEST SETİ DEĞERLENDİRMESİ\n');
    fprintf('%s\n', repmat('=', 1, 70));
    
    if nargin < 3
        outputDir = 'results';
    end
    if ~isfolder(outputDir), mkdir(outputDir); end
    
    X_test = data.X_test;
    y_test = double(data.y_test(:));
    numTest = length(y_test);
    
    fprintf('  Test seti: %d örnek\n', numTest);
    
    % ================================================================
    % 1. TAHMİN
    % ================================================================
    fprintf('\n[1/5] Tahminler alınıyor...\n');
    
    batchSize = 1024;
    numBatches = ceil(numTest / batchSize);
    y_pred_proba = zeros(numTest, 1, 'single');
    
    for b = 1:numBatches
        startIdx = (b - 1) * batchSize + 1;
        endIdx = min(b * batchSize, numTest);
        
        dlX = dlarray(X_test(:, :, startIdx:endIdx), 'CTB');
        if canUseGPU()
            dlX = gpuArray(dlX);
        end
        
        net = resetState(net); % Reset state before predicting to prevent batch size mismatch
        dlYPred = predict(net, dlX);
        preds = extractdata(squeeze(dlYPred));
        if canUseGPU()
            preds = gather(preds);
        end
        y_pred_proba(startIdx:endIdx) = preds(:);
    end
    
    y_scores = double(y_pred_proba);
    
    fprintf('  Tahmin aralığı: [%.6f, %.6f]\n', min(y_scores), max(y_scores));
    fprintf('  Tahmin ortalaması: %.6f\n', mean(y_scores));
    
    % ================================================================
    % 2. ROC-AUC & PR-AUC
    % ================================================================
    fprintf('\n[2/5] AUC metrikleri hesaplanıyor...\n');
    
    % ROC Eğrisi
    [fpr, tpr, thresholds_roc, rocAUC] = perfcurve(y_test, y_scores, 1);
    fprintf('  ROC-AUC: %.4f\n', rocAUC);
    
    % PR Eğrisi
    [recall_curve, precision_curve, thresholds_pr] = perfcurve(y_test, y_scores, 1, ...
        'XCrit', 'reca', 'YCrit', 'prec');
    
    % PR-AUC (trapez yöntemiyle)
    validIdx = ~isnan(precision_curve) & ~isnan(recall_curve);
    prAUC = abs(trapz(recall_curve(validIdx), precision_curve(validIdx)));
    fprintf('  PR-AUC:  %.4f\n', prAUC);
    
    % ================================================================
    % 3. OPTİMAL THRESHOLD (F1-Score Maximizasyonu)
    % ================================================================
    fprintf('\n[3/5] Optimal threshold aranıyor...\n');
    
    thresholds = 0.05:0.01:0.95;
    f1_scores = zeros(size(thresholds));
    
    for i = 1:length(thresholds)
        t = thresholds(i);
        y_pred_bin = double(y_scores >= t);
        
        tp = sum(y_pred_bin == 1 & y_test == 1);
        fp = sum(y_pred_bin == 1 & y_test == 0);
        fn = sum(y_pred_bin == 0 & y_test == 1);
        
        prec = tp / max(tp + fp, 1);
        rec  = tp / max(tp + fn, 1);
        
        if prec + rec > 0
            f1_scores(i) = 2 * prec * rec / (prec + rec);
        end
    end
    
    [bestF1, bestIdx] = max(f1_scores);
    bestThreshold = thresholds(bestIdx);
    fprintf('  Optimal Threshold: %.2f (F1=%.4f)\n', bestThreshold, bestF1);
    
    % ================================================================
    % 4. CONFUSION MATRIX & METRİKLER
    % ================================================================
    fprintf('\n[4/5] Confusion Matrix hesaplanıyor...\n');
    
    y_pred_binary = double(y_scores >= bestThreshold);
    
    % Manuel hesaplama (daha güvenilir)
    TP = sum(y_pred_binary == 1 & y_test == 1);
    TN = sum(y_pred_binary == 0 & y_test == 0);
    FP = sum(y_pred_binary == 1 & y_test == 0);
    FN = sum(y_pred_binary == 0 & y_test == 1);
    
    sensitivity = TP / max(TP + FN, 1);  % = Recall
    specificity = TN / max(TN + FP, 1);
    precision   = TP / max(TP + FP, 1);
    accuracy    = (TP + TN) / numTest;
    f1Score     = bestF1;
    
    fprintf('\n  ┌─────────────────────────────────────────┐\n');
    fprintf('  │         CONFUSION MATRIX                  │\n');
    fprintf('  ├─────────────────────────────────────────┤\n');
    fprintf('  │  TN: %-8d  │  FP: %-8d            │\n', TN, FP);
    fprintf('  │  FN: %-8d  │  TP: %-8d            │\n', FN, TP);
    fprintf('  └─────────────────────────────────────────┘\n');
    
    % ================================================================
    % 5. SONUÇ TABLOSU
    % ================================================================
    fprintf('\n[5/5] Sonuçlar özetleniyor...\n');
    
    fprintf('\n  %-20s │ %-10s\n', 'Metrik', 'Değer');
    fprintf('  %s\n', repmat('─', 1, 35));
    fprintf('  %-20s │ %.4f\n', 'ROC-AUC', rocAUC);
    fprintf('  %-20s │ %.4f\n', 'PR-AUC', prAUC);
    fprintf('  %-20s │ %.4f\n', 'F1-Score', f1Score);
    fprintf('  %-20s │ %.4f\n', 'Accuracy', accuracy);
    fprintf('  %-20s │ %.4f\n', 'Sensitivity', sensitivity);
    fprintf('  %-20s │ %.4f\n', 'Specificity', specificity);
    fprintf('  %-20s │ %.4f\n', 'Precision', precision);
    fprintf('  %-20s │ %.2f\n', 'Threshold', bestThreshold);
    
    % ================================================================
    % SONUÇLARI STRUCT'A KAYDET
    % ================================================================
    
    results.rocAUC       = rocAUC;
    results.prAUC        = prAUC;
    results.f1Score      = f1Score;
    results.accuracy     = accuracy;
    results.sensitivity  = sensitivity;
    results.specificity  = specificity;
    results.precision    = precision;
    results.bestThreshold = bestThreshold;
    results.TP = TP;
    results.TN = TN;
    results.FP = FP;
    results.FN = FN;
    results.y_scores     = y_scores;
    results.y_test       = y_test;
    results.f1_curve.thresholds = thresholds;
    results.f1_curve.scores     = f1_scores;
    results.roc_curve.fpr        = fpr;
    results.roc_curve.tpr        = tpr;
    results.pr_curve.recall      = recall_curve;
    results.pr_curve.precision   = precision_curve;
    
    % MAT dosyasına kaydet
    save(fullfile(outputDir, 'test_results.mat'), 'results');
    fprintf('\n  [OK] Sonuçlar kaydedildi: %s\n', fullfile(outputDir, 'test_results.mat'));
    
    fprintf('%s\n', repmat('=', 1, 70));
end
