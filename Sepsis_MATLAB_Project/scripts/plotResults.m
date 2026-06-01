function plotResults(results, history, outputDir)
% PLOTRESULTS Profesyonel kalitede eğitim ve değerlendirme grafikleri
%   plotResults(results, history, outputDir)
%
%   Oluşturulan Grafikler:
%     1. Eğitim Geçmişi (Loss & AUC vs Epoch)
%     2. ROC Eğrisi
%     3. Precision-Recall Eğrisi
%     4. Confusion Matrix Heatmap
%     5. F1-Score vs Threshold
%     6. Tahmin Dağılımı (Histogram)
%
%   Tüm grafikler 300 DPI ile PNG olarak kaydedilir.

    if nargin < 3
        outputDir = 'results/figures';
    end
    if ~isfolder(outputDir), mkdir(outputDir); end
    
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('GRAFİK OLUŞTURMA\n');
    fprintf('%s\n', repmat('=', 1, 60));
    
    % Renk paleti (profesyonel)
    colors.blue   = [0.2, 0.4, 0.8];
    colors.red    = [0.9, 0.2, 0.2];
    colors.green  = [0.2, 0.7, 0.3];
    colors.purple = [0.6, 0.2, 0.8];
    colors.orange = [0.95, 0.5, 0.1];
    colors.gold   = [0.85, 0.65, 0.13];
    colors.dark   = [0.15, 0.15, 0.2];
    
    % ================================================================
    % 1. EĞİTİM GEÇMİŞİ
    % ================================================================
    fprintf('  [1/6] Eğitim Geçmişi...\n');
    
    fig1 = figure('Name', 'Eğitim Geçmişi', 'Position', [50 50 1400 500], ...
        'Color', 'white');
    
    epochs = 1:length(history.trainLoss);
    
    % Loss
    subplot(1, 3, 1);
    plot(epochs, history.trainLoss, '-', 'Color', colors.blue, 'LineWidth', 2.5);
    hold on;
    plot(epochs, history.valLoss, '-', 'Color', colors.red, 'LineWidth', 2.5);
    if isfield(history, 'bestEpoch')
        xline(history.bestEpoch, '--', 'Color', colors.gold, 'LineWidth', 1.5, ...
            'Label', sprintf('En İyi (Epoch %d)', history.bestEpoch));
    end
    hold off;
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Focal Loss', 'FontSize', 12, 'FontWeight', 'bold');
    title('Kayıp Fonksiyonu', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Eğitim', 'Doğrulama', 'Location', 'northeast', 'FontSize', 11);
    grid on; grid minor;
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    
    % AUC
    subplot(1, 3, 2);
    plot(epochs, history.valAUC, '-', 'Color', colors.green, 'LineWidth', 2.5);
    hold on;
    plot(epochs, history.valPR_AUC, '-', 'Color', colors.purple, 'LineWidth', 2.5);
    if isfield(history, 'bestEpoch')
        xline(history.bestEpoch, '--', 'Color', colors.gold, 'LineWidth', 1.5);
    end
    hold off;
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('AUC Skoru', 'FontSize', 12, 'FontWeight', 'bold');
    title('Doğrulama AUC', 'FontSize', 14, 'FontWeight', 'bold');
    legend('ROC-AUC', 'PR-AUC', 'Location', 'southeast', 'FontSize', 11);
    grid on; grid minor;
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    ylim([0 1]);
    
    % Learning Rate
    subplot(1, 3, 3);
    semilogy(epochs, history.learnRate, '-', 'Color', colors.orange, 'LineWidth', 2.5);
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Öğrenme Oranı (log)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Öğrenme Oranı Zamanlaması', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    
    sgtitle('Sepsis Tahmin Modeli — Eğitim Geçmişi', ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig1, fullfile(outputDir, 'training_history.png'), 'Resolution', 300);
    savefig(fig1, fullfile(outputDir, 'training_history.fig'));
    
    % ================================================================
    % 2. ROC EĞRİSİ
    % ================================================================
    fprintf('  [2/6] ROC Eğrisi...\n');
    
    fig2 = figure('Name', 'ROC Eğrisi', 'Position', [100 100 700 600], ...
        'Color', 'white');
    
    fill([results.roc_curve.fpr; 1; 0], [results.roc_curve.tpr; 0; 0], ...
        colors.blue, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    hold on;
    plot(results.roc_curve.fpr, results.roc_curve.tpr, '-', ...
        'Color', colors.blue, 'LineWidth', 3);
    plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
    hold off;
    
    xlabel('Yanlış Pozitif Oranı (FPR)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Doğru Pozitif Oranı (TPR)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('ROC Eğrisi (AUC = %.4f)', results.rocAUC), ...
        'FontSize', 15, 'FontWeight', 'bold');
    legend(sprintf('Model (AUC=%.4f)', results.rocAUC), 'Rastgele', ...
        'Location', 'southeast', 'FontSize', 12);
    grid on; grid minor;
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    xlim([0 1]); ylim([0 1]);
    axis square;
    
    exportgraphics(fig2, fullfile(outputDir, 'roc_curve.png'), 'Resolution', 300);
    
    % ================================================================
    % 3. PRECISION-RECALL EĞRİSİ
    % ================================================================
    fprintf('  [3/6] Precision-Recall Eğrisi...\n');
    
    fig3 = figure('Name', 'PR Eğrisi', 'Position', [150 100 700 600], ...
        'Color', 'white');
    
    validIdx = ~isnan(results.pr_curve.precision) & ~isnan(results.pr_curve.recall);
    rec = results.pr_curve.recall(validIdx);
    prec = results.pr_curve.precision(validIdx);
    
    fill([rec; 0], [prec; 0], colors.purple, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    hold on;
    plot(rec, prec, '-', 'Color', colors.purple, 'LineWidth', 3);
    
    % Baseline (pozitif sınıf oranı)
    baseline = sum(results.y_test == 1) / length(results.y_test);
    yline(baseline, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
        'Label', sprintf('Baseline (%.2f%%)', baseline*100));
    hold off;
    
    xlabel('Recall (Duyarlılık)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Precision (Kesinlik)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Precision-Recall Eğrisi (AUC = %.4f)', results.prAUC), ...
        'FontSize', 15, 'FontWeight', 'bold');
    legend(sprintf('Model (PR-AUC=%.4f)', results.prAUC), 'Location', 'northeast', 'FontSize', 12);
    grid on; grid minor;
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    xlim([0 1]); ylim([0 1]);
    
    exportgraphics(fig3, fullfile(outputDir, 'pr_curve.png'), 'Resolution', 300);
    
    % ================================================================
    % 4. CONFUSION MATRIX HEATMAP
    % ================================================================
    fprintf('  [4/6] Confusion Matrix...\n');
    
    fig4 = figure('Name', 'Confusion Matrix', 'Position', [200 100 650 550], ...
        'Color', 'white');
    
    cm = [results.TN, results.FP; results.FN, results.TP];
    cmNorm = cm ./ sum(cm, 2) .* 100;  % Satır bazlı yüzde
    
    imagesc(cmNorm);
    colormap(flipud(hot));
    colorbar('FontSize', 11);
    
    % Hücre değerlerini yaz
    labels = {'Negatif (0)', 'Pozitif (1)'};
    for i = 1:2
        for j = 1:2
            if cmNorm(i,j) > 50
                txtColor = 'white';
            else
                txtColor = 'black';
            end
            text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), cmNorm(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, ...
                'FontWeight', 'bold', 'Color', txtColor);
        end
    end
    
    set(gca, 'XTick', 1:2, 'XTickLabel', labels, 'FontSize', 12);
    set(gca, 'YTick', 1:2, 'YTickLabel', labels, 'FontSize', 12);
    xlabel('Tahmin Edilen', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Gerçek', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Confusion Matrix (Threshold=%.2f)', results.bestThreshold), ...
        'FontSize', 15, 'FontWeight', 'bold');
    
    exportgraphics(fig4, fullfile(outputDir, 'confusion_matrix.png'), 'Resolution', 300);
    
    % ================================================================
    % 5. F1-SCORE vs THRESHOLD
    % ================================================================
    fprintf('  [5/6] F1-Score vs Threshold...\n');
    
    fig5 = figure('Name', 'F1 vs Threshold', 'Position', [250 100 700 500], ...
        'Color', 'white');
    
    plot(results.f1_curve.thresholds, results.f1_curve.scores, '-', ...
        'Color', colors.green, 'LineWidth', 2.5);
    hold on;
    plot(results.bestThreshold, results.f1Score, 'o', ...
        'MarkerSize', 12, 'MarkerFaceColor', colors.red, ...
        'MarkerEdgeColor', 'white', 'LineWidth', 2);
    xline(results.bestThreshold, '--', 'Color', colors.red, 'LineWidth', 1.5, ...
        'Label', sprintf('Optimal: %.2f', results.bestThreshold));
    hold off;
    
    xlabel('Threshold', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('F1-Score', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('F1-Score vs Threshold (Max F1=%.4f)', results.f1Score), ...
        'FontSize', 15, 'FontWeight', 'bold');
    grid on; grid minor;
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    xlim([0 1]);
    
    exportgraphics(fig5, fullfile(outputDir, 'f1_vs_threshold.png'), 'Resolution', 300);
    
    % ================================================================
    % 6. TAHMİN DAĞILIMI
    % ================================================================
    fprintf('  [6/6] Tahmin Dağılımı...\n');
    
    fig6 = figure('Name', 'Tahmin Dağılımı', 'Position', [300 100 800 500], ...
        'Color', 'white');
    
    negScores = results.y_scores(results.y_test == 0);
    posScores = results.y_scores(results.y_test == 1);
    
    edges = linspace(0, 1, 51);
    
    histogram(negScores, edges, 'FaceColor', colors.blue, 'FaceAlpha', 0.6, ...
        'EdgeColor', 'none', 'Normalization', 'probability');
    hold on;
    histogram(posScores, edges, 'FaceColor', colors.red, 'FaceAlpha', 0.6, ...
        'EdgeColor', 'none', 'Normalization', 'probability');
    xline(results.bestThreshold, '--', 'Color', colors.gold, 'LineWidth', 2.5, ...
        'Label', sprintf('Threshold: %.2f', results.bestThreshold));
    hold off;
    
    xlabel('Tahmin Olasılığı', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Normalize Frekans', 'FontSize', 13, 'FontWeight', 'bold');
    title('Sınıf Bazlı Tahmin Dağılımı', 'FontSize', 15, 'FontWeight', 'bold');
    legend('Negatif (Sepsis Yok)', 'Pozitif (Sepsis)', 'Threshold', ...
        'Location', 'northeast', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    
    exportgraphics(fig6, fullfile(outputDir, 'prediction_distribution.png'), 'Resolution', 300);
    
    % ================================================================
    % ÖZET
    % ================================================================
    fprintf('\n  [OK] 6 grafik oluşturuldu ve kaydedildi:\n');
    fprintf('       → %s\n', outputDir);
    fprintf('%s\n', repmat('=', 1, 60));
end
