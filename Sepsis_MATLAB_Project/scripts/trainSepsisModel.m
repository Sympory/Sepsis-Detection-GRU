function [trainedNet, history] = trainSepsisModel(net, data, options, resumeState)
% TRAINSEPSISMODEL Custom training loop ile Sepsis modelini eğitir
%   [trainedNet, history] = trainSepsisModel(net, data, options)
%   [trainedNet, history] = trainSepsisModel(net, data, options, resumeState)
%
%   Python'daki eğitim döngüsünün MATLAB karşılığı.
%   Adam optimizer, Focal Loss, Early Stopping, LR Scheduling.
%
%   ★ CHECKPOINT SİSTEMİ: Her epoch sonunda otomatik kaydeder.
%     Eğitim kesintiye uğrarsa restoreCheckpoint ile devam edin.
%
%   Girdi:
%     net     - dlnetwork (buildEnhancedModel çıktısı)
%     data    - struct (loadSepsisData çıktısı, feature eng. sonrası)
%     options - struct (opsiyonel eğitim parametreleri)
%       .maxEpochs      (60)
%       .miniBatchSize   (512)
%       .initialLearnRate(0.001)
%       .focalGamma      (2.0)
%       .focalAlpha      (0.25)
%       .patience        (10)    - Early stopping patience
%       .lrPatience      (5)     - LR azaltma patience
%       .lrFactor        (0.5)   - LR azaltma faktörü
%       .minLR           (1e-6)  - Minimum learning rate
%       .useGPU          (true)  - GPU kullan
%       .outputDir       ('results')
%       .checkpointEvery (1)     - Kaç epoch'ta bir checkpoint kaydet
%       .trainAUCEvery   (5)     - Kaç epoch'ta bir gerçek train AUC hesapla
%     resumeState - (opsiyonel) restoreCheckpoint çıktısı, devam için

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('MODEL EĞİTİMİ (Custom Training Loop + Checkpoint)\n');
    fprintf('%s\n', repmat('=', 1, 70));

    % ================================================================
    % VARSAYILAN PARAMETRELER
    % ================================================================
    if nargin < 3, options = struct(); end
    if nargin < 4, resumeState = []; end
    
    opts.maxEpochs       = getOpt(options, 'maxEpochs', 60);
    opts.miniBatchSize   = getOpt(options, 'miniBatchSize', 512);
    opts.initialLearnRate= getOpt(options, 'initialLearnRate', 0.001);
    opts.focalGamma      = getOpt(options, 'focalGamma', 2.0);
    opts.focalAlpha      = getOpt(options, 'focalAlpha', 0.25);
    opts.patience        = getOpt(options, 'patience', 10);
    opts.lrPatience      = getOpt(options, 'lrPatience', 5);
    opts.lrFactor        = getOpt(options, 'lrFactor', 0.5);
    opts.minLR           = getOpt(options, 'minLR', 1e-6);
    opts.useGPU          = getOpt(options, 'useGPU', canUseGPU());
    opts.outputDir       = getOpt(options, 'outputDir', 'results');
    opts.checkpointEvery = getOpt(options, 'checkpointEvery', 1);
    opts.trainAUCEvery   = getOpt(options, 'trainAUCEvery', 5);
    
    % Checkpoint dizini
    checkpointDir = fullfile(opts.outputDir, 'checkpoints');
    if ~isfolder(checkpointDir), mkdir(checkpointDir); end
    if ~isfolder(opts.outputDir), mkdir(opts.outputDir); end
    
    fprintf('\nEğitim Parametreleri:\n');
    fprintf('  Max Epoch:       %d\n', opts.maxEpochs);
    fprintf('  Batch Size:      %d\n', opts.miniBatchSize);
    fprintf('  Learning Rate:   %.4f\n', opts.initialLearnRate);
    fprintf('  Focal Loss:      gamma=%.1f, alpha=%.2f\n', opts.focalGamma, opts.focalAlpha);
    fprintf('  Early Stopping:  patience=%d\n', opts.patience);
    fprintf('  LR Scheduling:   patience=%d, factor=%.1f\n', opts.lrPatience, opts.lrFactor);
    fprintf('  GPU:             %s\n', string(opts.useGPU));
    fprintf('  Checkpoint:      Her %d epoch\n', opts.checkpointEvery);
    fprintf('  Train AUC:       Her %d epoch (gerçek hesaplama)\n', opts.trainAUCEvery);
    
    % ================================================================
    % VERİ HAZIRLAMA
    % ================================================================
    
    X_train = data.X_train;
    Y_train = data.y_train;
    X_val   = data.X_val;
    Y_val   = data.y_val;
    
    numTrain = size(X_train, 3);
    numBatches = ceil(numTrain / opts.miniBatchSize);
    
    fprintf('\n  Train: %d örnek (%d batch)\n', numTrain, numBatches);
    fprintf('  Val:   %d örnek\n', size(X_val, 3));
    
    % ================================================================
    % BAŞLANGIÇ DURUMUNU AYARLA (Yeni veya Resume)
    % ================================================================
    
    if ~isempty(resumeState)
        % --- CHECKPOINT'TEN DEVAM ---
        fprintf('\n   CHECKPOINT''TEN DEVAM EDİLİYOR (Epoch %d)\n', resumeState.epoch);
        
        avgGrad         = resumeState.avgGrad;
        avgSqGrad       = resumeState.avgSqGrad;
        learnRate       = resumeState.learnRate;
        iteration       = resumeState.iteration;
        startEpoch      = resumeState.epoch + 1;
        bestValPR_AUC   = resumeState.bestValPR_AUC;
        bestValAUC      = 0;
        bestNet         = resumeState.bestNet;
        bestEpoch       = resumeState.bestEpoch;
        
        % History'yi genişlet
        prevHistory     = resumeState.history;
        prevEpochs      = numel(prevHistory.trainLoss);
        
        history.trainLoss   = [prevHistory.trainLoss;   zeros(opts.maxEpochs, 1)];
        history.valLoss     = [prevHistory.valLoss;     zeros(opts.maxEpochs, 1)];
        history.trainAUC    = [prevHistory.trainAUC;    zeros(opts.maxEpochs, 1)];
        history.valAUC      = [prevHistory.valAUC;      zeros(opts.maxEpochs, 1)];
        history.valPR_AUC   = [prevHistory.valPR_AUC;   zeros(opts.maxEpochs, 1)];
        history.learnRate   = [prevHistory.learnRate;   zeros(opts.maxEpochs, 1)];
        
        % Early stopping sayaçlarını kur
        epochsNoImprove = resumeState.epoch - resumeState.bestEpoch;
        lrNoImprove     = epochsNoImprove;  % Yaklaşık
        
        fprintf('  Devam: Epoch %d''den başlanıyor\n', startEpoch);
        fprintf('  Best PR-AUC: %.4f (Epoch %d)\n', bestValPR_AUC, bestEpoch);
    else
        % --- YENİ EĞİTİM ---
        avgGrad         = [];
        avgSqGrad       = [];
        learnRate       = opts.initialLearnRate;
        iteration       = 0;
        startEpoch      = 1;
        bestValPR_AUC   = 0;
        bestValAUC      = 0;
        bestNet         = net;
        bestEpoch       = 0;
        prevEpochs      = 0;
        
        history.trainLoss   = zeros(opts.maxEpochs, 1);
        history.valLoss     = zeros(opts.maxEpochs, 1);
        history.trainAUC    = zeros(opts.maxEpochs, 1);
        history.valAUC      = zeros(opts.maxEpochs, 1);
        history.valPR_AUC   = zeros(opts.maxEpochs, 1);
        history.learnRate   = zeros(opts.maxEpochs, 1);
        
        epochsNoImprove = 0;
        lrNoImprove     = 0;
    end
    
    lastRealTrainAUC = 0.5;  % Son gerçek train AUC değeri
    
    % ================================================================
    % EĞİTİM İLERLEME GRAFİĞİ
    % ================================================================
    
    fig = figure('Name', 'Eğitim İlerlemesi', 'Position', [100 100 1200 400]);
    
    % ================================================================
    % ANA EĞİTİM DÖNGÜSÜ
    % ================================================================
    
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('%-6s | %-10s | %-10s | %-10s | %-10s | %-10s | %-8s\n', ...
        'Epoch', 'Train Loss', 'Val Loss', 'Train AUC', 'Val AUC', 'Val PR-AUC', 'LR');
    fprintf('%s\n', repmat('-', 1, 80));
    
    trainStart = tic;
    
    for epoch = startEpoch:(startEpoch + opts.maxEpochs - 1)
        epochStart = tic;
        
        % History indeksi (önceki epoch'ları da hesaba kat)
        histIdx = prevEpochs + (epoch - startEpoch + 1);
        
        % Veriyi karıştır (shuffle)
        shuffleIdx = randperm(numTrain);
        X_train_shuffled = X_train(:, :, shuffleIdx);
        Y_train_shuffled = Y_train(:, shuffleIdx);
        
        epochLoss = 0;
        
        % Train AUC için tahminleri biriktir (gerçek hesaplama epoch'larında)
        computeRealTrainAUC = (mod(epoch, opts.trainAUCEvery) == 0) || (epoch == startEpoch);
        if computeRealTrainAUC
            allTrainPreds = zeros(1, numTrain, 'single');
        end
        
        % --- Mini-Batch Döngüsü ---
        for batch = 1:numBatches
            iteration = iteration + 1;
            
            % Batch indeksleri
            startIdx = (batch - 1) * opts.miniBatchSize + 1;
            endIdx = min(batch * opts.miniBatchSize, numTrain);
            
            % Batch verisi
            XBatch = X_train_shuffled(:, :, startIdx:endIdx);
            YBatch = Y_train_shuffled(:, startIdx:endIdx);
            
            % dlarray'e çevir
            dlX = dlarray(XBatch, 'CTB');
            dlY = dlarray(YBatch, 'CB');
            
            % GPU'ya taşı
            if opts.useGPU
                dlX = gpuArray(dlX);
                dlY = gpuArray(dlY);
            end
            
            % Reset state for independent sequences in this batch
            net = resetState(net);
            
            % Gradient hesapla
            [loss, gradients, state, dlYPred] = dlfeval(@modelLoss, net, dlX, dlY, ...
                opts.focalGamma, opts.focalAlpha);
            net.State = state;
            
            % Adam update
            [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, ...
                avgGrad, avgSqGrad, iteration, learnRate);
            
            epochLoss = epochLoss + double(extractdata(loss));
            
            % Train AUC için tahminleri sakla
            if computeRealTrainAUC
                preds = extractdata(squeeze(dlYPred));
                if opts.useGPU, preds = gather(preds); end
                if iscolumn(preds), preds = preds'; end
                allTrainPreds(startIdx:endIdx) = preds;
            end
        end
        
        % Epoch ortalaması
        avgTrainLoss = epochLoss / numBatches;
        
        % --- Train AUC Hesaplama ---
        if computeRealTrainAUC
            try
                trainLabels = double(gather(Y_train_shuffled(:)));
                trainScores = double(allTrainPreds(:));
                [~, ~, ~, trainAUC] = perfcurve(trainLabels, trainScores, 1);
                lastRealTrainAUC = trainAUC;
            catch
                trainAUC = lastRealTrainAUC;
            end
        else
            % Ara epoch'larda son gerçek değeri kullan
            trainAUC = lastRealTrainAUC;
        end
        
        % --- Validation Değerlendirme ---
        [valLoss, valAUC, valPR_AUC] = evaluateOnSet(net, X_val, Y_val, ...
            opts.miniBatchSize, opts.focalGamma, opts.focalAlpha, opts.useGPU);
        
        % Geçmişe kaydet
        history.trainLoss(histIdx)  = avgTrainLoss;
        history.valLoss(histIdx)    = valLoss;
        history.trainAUC(histIdx)   = trainAUC;
        history.valAUC(histIdx)     = valAUC;
        history.valPR_AUC(histIdx)  = valPR_AUC;
        history.learnRate(histIdx)  = learnRate;
        
        % Epoch süresini hesapla
        epochTime = toc(epochStart);
        
        % Konsola yazdır
        aucLabel = '';
        if computeRealTrainAUC
            aucLabel = ' (R)';  % (R) = gerçek hesaplama
        end
        fprintf('%-6d | %-10.4f | %-10.4f | %-10.4f%s | %-10.4f | %-10.4f | %-8.6f', ...
            epoch, avgTrainLoss, valLoss, trainAUC, aucLabel, valAUC, valPR_AUC, learnRate);
        
        % --- Early Stopping Kontrolü ---
        if valPR_AUC > bestValPR_AUC
            bestValPR_AUC = valPR_AUC;
            bestValAUC = valAUC;
            bestNet = net;
            bestEpoch = epoch;
            epochsNoImprove = 0;
            lrNoImprove = 0;
            fprintf(' ★ Best! (%.0fs)\n', epochTime);
            
            % En iyi modeli kaydet
            save(fullfile(opts.outputDir, 'best_model.mat'), 'net', 'epoch', 'valAUC', 'valPR_AUC');
        else
            epochsNoImprove = epochsNoImprove + 1;
            lrNoImprove = lrNoImprove + 1;
            fprintf(' (%.0fs)\n', epochTime);
        end
        
        % --- CHECKPOINT KAYDET ---
        if mod(epoch, opts.checkpointEvery) == 0
            saveCheckpoint(checkpointDir, epoch, net, avgGrad, avgSqGrad, ...
                iteration, learnRate, history, histIdx, bestNet, bestValPR_AUC, bestEpoch);
        end
        
        % --- LR Scheduling (ReduceLROnPlateau) ---
        if lrNoImprove >= opts.lrPatience
            newLR = max(learnRate * opts.lrFactor, opts.minLR);
            if newLR < learnRate
                fprintf('  → Learning Rate azaltıldı: %.6f → %.6f\n', learnRate, newLR);
                learnRate = newLR;
                lrNoImprove = 0;
            end
        end
        
        % --- Early Stopping ---
        if epochsNoImprove >= opts.patience
            fprintf('\n  [EARLY STOPPING] %d epoch boyunca iyileşme olmadı.\n', opts.patience);
            break;
        end
        
        % --- Canlı Grafik Güncelle ---
        updateTrainingPlot(fig, history, histIdx);
    end
    
    trainElapsed = toc(trainStart);
    actualEpochs = histIdx;
    
    % Geçmişi kırp
    history.trainLoss  = history.trainLoss(1:actualEpochs);
    history.valLoss    = history.valLoss(1:actualEpochs);
    history.trainAUC   = history.trainAUC(1:actualEpochs);
    history.valAUC     = history.valAUC(1:actualEpochs);
    history.valPR_AUC  = history.valPR_AUC(1:actualEpochs);
    history.learnRate  = history.learnRate(1:actualEpochs);
    history.bestEpoch  = bestEpoch;
    history.trainingTime = trainElapsed;
    
    % En iyi modeli döndür
    trainedNet = bestNet;
    
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('EĞİTİM TAMAMLANDI\n');
    fprintf('  Süre:         %.1f dakika\n', trainElapsed/60);
    fprintf('  Epoch:        %d → %d\n', startEpoch, epoch);
    fprintf('  En iyi epoch: %d\n', bestEpoch);
    fprintf('  Best Val AUC: %.4f\n', bestValAUC);
    fprintf('  Best PR-AUC:  %.4f\n', bestValPR_AUC);
    fprintf('  Checkpoint:   %s\n', checkpointDir);
    fprintf('%s\n', repmat('=', 1, 70));
    
    % Son checkpoint'i kaydet
    saveCheckpoint(checkpointDir, epoch, net, avgGrad, avgSqGrad, ...
        iteration, learnRate, history, actualEpochs, bestNet, bestValPR_AUC, bestEpoch);
    fprintf('  [OK] Son checkpoint kaydedildi.\n');
    
    % Eğitim grafiğini kaydet
    if ~isgraphics(fig) || ~isvalid(fig)
        fig = figure('Name', 'Eğitim İlerlemesi', 'Position', [100 100 1200 400], 'Visible', 'off');
        updateTrainingPlot(fig, history, actualEpochs);
    end
    saveas(fig, fullfile(opts.outputDir, 'training_progress.png'));
    saveas(fig, fullfile(opts.outputDir, 'training_progress.fig'));
    if isgraphics(fig) && isvalid(fig)
        close(fig);
    end
    
    % History kaydet
    save(fullfile(opts.outputDir, 'training_history.mat'), 'history');
end


% ================================================================
% CHECKPOINT KAYDETME
% ================================================================

function saveCheckpoint(checkpointDir, epoch, net, avgGrad, avgSqGrad, ...
    iteration, learnRate, history, histIdx, bestNet, bestValPR_AUC, bestEpoch)
% Her epoch sonunda eğitim durumunu kaydeder

    cpFile = fullfile(checkpointDir, sprintf('checkpoint_epoch_%d.mat', epoch));

    % History'yi kırp (sadece dolu kısım)
    cpHistory.trainLoss = history.trainLoss(1:histIdx);
    cpHistory.valLoss   = history.valLoss(1:histIdx);
    cpHistory.trainAUC  = history.trainAUC(1:histIdx);
    cpHistory.valAUC    = history.valAUC(1:histIdx);
    cpHistory.valPR_AUC = history.valPR_AUC(1:histIdx);
    cpHistory.learnRate = history.learnRate(1:histIdx);

    save(cpFile, 'net', 'avgGrad', 'avgSqGrad', 'iteration', ...
        'learnRate', 'history', 'bestNet', 'bestValPR_AUC', 'bestEpoch', 'epoch', ...
        '-v7.3');
    
    % history alanını kırpılmış versiyonla değiştir
    save(cpFile, 'net', 'avgGrad', 'avgSqGrad', 'iteration', ...
        'learnRate', 'bestNet', 'bestValPR_AUC', 'bestEpoch', 'epoch', ...
        '-v7.3');
    % Kırpılmış history'yi ayrı kaydet
    history = cpHistory; %#ok<NASGU>
    save(cpFile, 'history', '-append');

    % Eski checkpoint'leri temizle (son 5'i tut + best epoch)
    cpFiles = dir(fullfile(checkpointDir, 'checkpoint_epoch_*.mat'));
    if numel(cpFiles) > 6
        epochs = zeros(numel(cpFiles), 1);
        for i = 1:numel(cpFiles)
            tokens = regexp(cpFiles(i).name, 'checkpoint_epoch_(\d+)', 'tokens');
            if ~isempty(tokens)
                epochs(i) = str2double(tokens{1}{1});
            end
        end
        [sortedEpochs, sortIdx] = sort(epochs);
        
        % Son 5 + best epoch'u koru
        keepEpochs = sortedEpochs(max(1, end-4):end);
        if ~ismember(bestEpoch, keepEpochs) && bestEpoch > 0
            keepEpochs = [keepEpochs; bestEpoch];
        end
        
        for i = 1:numel(cpFiles)
            if ~ismember(epochs(sortIdx(i)), keepEpochs)
                delete(fullfile(cpFiles(sortIdx(i)).folder, cpFiles(sortIdx(i)).name));
            end
        end
    end
end


% ================================================================
% YARDIMCI FONKSİYONLAR
% ================================================================

function [loss, gradients, state, dlYPred] = modelLoss(net, dlX, dlY, gamma, alpha)
% Model kaybını ve gradientleri hesaplar
    [dlYPred, state] = forward(net, dlX);
    
    % Format etiketlerini kaldır
    dlYPred = stripdims(dlYPred);
    dlY = stripdims(dlY);
    
    % Batch boyutunu al ve her ikisini de [1, batchSize] yap
    batchSize = numel(dlY);
    dlYPred = reshape(dlYPred, 1, batchSize);
    dlY = reshape(dlY, 1, batchSize);
    
    loss = focalLoss(dlYPred, dlY, gamma, alpha);
    gradients = dlgradient(loss, net.Learnables);
end


function [avgLoss, rocAUC, prAUC] = evaluateOnSet(net, X, Y, batchSize, gamma, alpha, useGPU)
% Bir veri seti üzerinde loss ve AUC hesaplar
    numSamples = size(X, 3);
    numBatches = ceil(numSamples / batchSize);
    
    totalLoss = 0;
    allPreds = zeros(1, numSamples, 'single');
    
    for b = 1:numBatches
        startIdx = (b - 1) * batchSize + 1;
        endIdx = min(b * batchSize, numSamples);
        
        dlX = dlarray(X(:, :, startIdx:endIdx), 'CTB');
        if useGPU
            dlX = gpuArray(dlX);
        end
        
        net = resetState(net); % Reset state before predicting to prevent batch size mismatch
        dlYPred = predict(net, dlX);
        preds = extractdata(squeeze(dlYPred));
        if useGPU
            preds = gather(preds);
        end
        
        % Boyut düzeltme
        if iscolumn(preds)
            preds = preds';
        end
        allPreds(startIdx:endIdx) = preds;
        
        % Batch loss
        dlY = dlarray(Y(:, startIdx:endIdx), 'CB');
        if useGPU, dlY = gpuArray(dlY); end
        bLoss = focalLoss(dlarray(preds, 'CB'), dlarray(gather(Y(:, startIdx:endIdx)), 'CB'), gamma, alpha);
        totalLoss = totalLoss + double(extractdata(bLoss));
    end
    
    avgLoss = totalLoss / numBatches;
    
    % ROC-AUC hesapla
    labels = double(gather(Y(:)));
    scores = double(allPreds(:));
    
    try
        [~, ~, ~, rocAUC] = perfcurve(labels, scores, 1);
    catch
        rocAUC = 0.5;
    end
    
    % PR-AUC hesapla
    try
        [rec, prec, ~] = perfcurve(labels, scores, 1, 'XCrit', 'reca', 'YCrit', 'prec');
        % NaN temizliği
        validIdx = ~isnan(prec) & ~isnan(rec);
        prAUC = trapz(rec(validIdx), prec(validIdx));
        prAUC = abs(prAUC); % Pozitif olmalı
    catch
        prAUC = 0;
    end
end


function updateTrainingPlot(fig, history, epoch)
% Eğitim grafiğini canlı günceller
    if ~isvalid(fig), return; end
    
    figure(fig);
    epochs = 1:epoch;
    
    % Loss grafiği
    subplot(1, 3, 1);
    plot(epochs, history.trainLoss(epochs), 'b-', 'LineWidth', 2); hold on;
    plot(epochs, history.valLoss(epochs), 'r-', 'LineWidth', 2); hold off;
    xlabel('Epoch'); ylabel('Focal Loss');
    title('Kayıp Fonksiyonu');
    legend('Train', 'Validation', 'Location', 'northeast');
    grid on;
    
    % AUC grafiği
    subplot(1, 3, 2);
    plot(epochs, history.trainAUC(epochs), 'b--', 'LineWidth', 1.5); hold on;
    plot(epochs, history.valAUC(epochs), 'g-', 'LineWidth', 2);
    plot(epochs, history.valPR_AUC(epochs), 'm-', 'LineWidth', 2); hold off;
    xlabel('Epoch'); ylabel('AUC');
    title('AUC Metrikleri');
    legend('Train ROC-AUC', 'Val ROC-AUC', 'Val PR-AUC', 'Location', 'southeast');
    grid on;
    
    % Learning Rate grafiği
    subplot(1, 3, 3);
    semilogy(epochs, history.learnRate(epochs), 'k-', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Learning Rate');
    title('Öğrenme Oranı');
    grid on;
    
    drawnow;
end


function val = getOpt(s, field, default)
% Struct'tan alan oku, yoksa varsayılan kullan
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
