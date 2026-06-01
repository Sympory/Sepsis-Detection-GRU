%% ========================================================================
%  SEPSİS ERKEN TAHMİN SİSTEMİ — ANA ÇALIŞTIRMA BETİĞİ (main.m)
%  ========================================================================
%  Enhanced v2 Mimari: Bi-LSTM + Self-Attention + Feature Engineering
%  
%  Bu dosya tüm pipeline'ı sırasıyla çalıştırır:
%    1. Veri yükleme
%    2. Feature Engineering (12 yeni özellik)
%    3. Model oluşturma
%    4. Eğitim (Custom Training Loop + Focal Loss)
%    5. Değerlendirme
%    6. Görselleştirme
%    7. Rapor oluşturma
%
%  ★ CHECKPOINT SİSTEMİ:
%    - Her epoch sonunda otomatik kaydeder
%    - Eğitim kesintiye uğrarsa kaldığınız yerden devam edebilirsiniz
%    - Pipeline durumu da her adım sonunda kaydedilir
%
%  Kullanım:
%    MATLAB'da bu dosyayı açıp "Run" butonuna basın.
%    Veya Command Window'da: >> main
%
%  Checkpoint'ten devam:
%    >> main   % Otomatik olarak sorar
%
%  Gereksinimler:
%    - MATLAB R2024a+ (R2025b önerilir)
%    - Deep Learning Toolbox
%    - Sepsis veri seti (data/sepsis_data.mat)
%      → Yoksa önce: python scripts/convert_npy_to_mat.py
%  ========================================================================

clc; close all;

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   SEPSİS ERKEN TAHMİN SİSTEMİ — MATLAB Deep Learning      ║\n');
fprintf('║   Enhanced v2: Bi-LSTM + Self-Attention + Checkpoint        ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('\nBaşlangıç: %s\n', datestr(now, 'dd.mm.yyyy HH:MM:SS'));

% Proje yollarını ayarla
projectRoot = fileparts(mfilename('fullpath'));
run(fullfile(projectRoot, 'startup.m'));

% Sonuç klasörleri
outputDir     = fullfile(projectRoot, 'results');
figureDir     = fullfile(outputDir, 'figures');
checkpointDir = fullfile(outputDir, 'checkpoints');
if ~isfolder(outputDir),     mkdir(outputDir); end
if ~isfolder(figureDir),     mkdir(figureDir); end
if ~isfolder(checkpointDir), mkdir(checkpointDir); end

% Pipeline state dosyası
pipelineStateFile = fullfile(outputDir, 'pipeline_state.mat');

%% ========================================================================
%  CHECKPOINT / DEVAM KONTROLÜ
%  ========================================================================

resumeMode = false;
resumeState = [];
startStep = 1;

if isfile(pipelineStateFile)
    fprintf('\n  ┌─────────────────────────────────────────────────────────┐\n');
    fprintf('  │  ★ Önceki bir çalışma tespit edildi!                    │\n');
    fprintf('  └─────────────────────────────────────────────────────────┘\n');
    
    prevState = load(pipelineStateFile);
    fprintf('  Son tamamlanan adım: %d/7 (%s)\n', prevState.lastStep, prevState.stepName);
    fprintf('  Tarih: %s\n', prevState.timestamp);
    
    fprintf('\n  Ne yapmak istiyorsunuz?\n');
    fprintf('  [1] Kaldığım yerden devam et\n');
    fprintf('  [2] Baştan başla (mevcut sonuçlar korunur)\n');
    fprintf('  [3] Baştan başla + yedek al\n');
    choice = input('  Seçiminiz (1/2/3): ', 's');
    
    switch choice
        case '1'
            resumeMode = true;
            startStep = prevState.lastStep + 1;
            fprintf('\n  → Adım %d''den devam ediliyor...\n', startStep);
            
            % Modeli ve sonuçları yükle (varsa)
            if isfield(prevState, 'net'),     net     = prevState.net; end
            if isfield(prevState, 'results'), results = prevState.results; end
            if isfield(prevState, 'history'), history = prevState.history; end
            
            % Veriyi tekrar yükle (pipeline state'te saklanmıyor, çok büyük)
            if startStep > 1
                fprintf('  → Veri tekrar yükleniyor...\n');
                dataPath = fullfile(projectRoot, 'data', 'sepsis_data.mat');
                data = loadSepsisData(dataPath);
            end
            if startStep > 2
                fprintf('  → Feature engineering tekrar uygulanıyor...\n');
                [data.X_train, newCols] = engineerFeatures(data.X_train, data.columnNames);
                [data.X_val, ~]         = engineerFeatures(data.X_val, data.columnNames);
                [data.X_test, ~]        = engineerFeatures(data.X_test, data.columnNames);
                data.columnNames = newCols;
                data.numFeatures = size(data.X_train, 1);
            end
            
            % net yerine trainedNet kullan (eğitim sonrası adımlarda)
            if startStep > 4 && isfield(prevState, 'net')
                trainedNet = prevState.net;
            end
            
        case '3'
            fprintf('\n  Yedek alınıyor...\n');
            backup_project(projectRoot);
            fprintf('  → Baştan başlanıyor...\n');
            
        otherwise
            fprintf('  → Baştan başlanıyor...\n');
    end
end

% Eğitim checkpoint kontrolü (ayrı)
cpFiles = dir(fullfile(checkpointDir, 'checkpoint_epoch_*.mat'));
if ~isempty(cpFiles) && startStep <= 4
    fprintf('\n  [BİLGİ] %d adet eğitim checkpoint''i mevcut.\n', numel(cpFiles));
    
    if startStep == 4 || (resumeMode && startStep <= 4)
        fprintf('  Checkpoint''ten devam etmek istiyor musunuz? (e/h): ');
        cpChoice = input('', 's');
        if strcmpi(cpChoice, 'e') || strcmpi(cpChoice, 'y')
            [net, resumeState] = restoreCheckpoint(checkpointDir);
        end
    end
end

%% ========================================================================
%  ADIM 1: VERİ YÜKLEME
%  ========================================================================

if startStep <= 1
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 1/7: VERİ YÜKLEME ━━━\n');
    dataPath = fullfile(projectRoot, 'data', 'sepsis_data.mat');
    
    if ~isfile(dataPath)
        fprintf('[HATA] Veri dosyası bulunamadı: %s\n', dataPath);
        fprintf('[BİLGİ] Önce Python dönüştürücüyü çalıştırın:\n');
        fprintf('        python scripts/convert_npy_to_mat.py\n');
        return;
    end
    
    data = loadSepsisData(dataPath);
    savePipelineState(pipelineStateFile, 1, 'Veri Yükleme', data);
end

%% ========================================================================
%  ADIM 2: FEATURE ENGINEERING (12 Yeni Özellik)
%  ========================================================================

if startStep <= 2
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 2/7: FEATURE ENGINEERING ━━━\n');
    
    [data.X_train, newCols] = engineerFeatures(data.X_train, data.columnNames);
    [data.X_val, ~]         = engineerFeatures(data.X_val, data.columnNames);
    [data.X_test, ~]        = engineerFeatures(data.X_test, data.columnNames);
    
    data.columnNames = newCols;
    data.numFeatures = size(data.X_train, 1);  % Güncelle
    
    fprintf('\n  Yeni özellik sayısı: %d (orijinal) + 12 (mühendislik) = %d\n', ...
        data.numFeatures - 12, data.numFeatures);
    
    savePipelineState(pipelineStateFile, 2, 'Feature Engineering', data);
end

%% ========================================================================
%  ADIM 3: MODEL OLUŞTURMA
%  ========================================================================

if startStep <= 3
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 3/7: MODEL OLUŞTURMA ━━━\n');
    
    numFeatures  = size(data.X_train, 1);  % Kanal/özellik sayısı
    numTimeSteps = size(data.X_train, 2);  % Zaman penceresi
    
    % Checkpoint'ten devam ediyorsak yeni model oluşturma
    if isempty(resumeState)
        net = buildEnhancedModel(numFeatures, numTimeSteps);
    else
        fprintf('  → Checkpoint''teki model kullanılıyor.\n');
    end
    
    savePipelineState(pipelineStateFile, 3, 'Model Oluşturma', data, net);
end

%% ========================================================================
%  ADIM 4: MODEL EĞİTİMİ
%  ========================================================================

if startStep <= 4
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 4/7: MODEL EĞİTİMİ ━━━\n');
    
    % Eğitim parametreleri
    trainOptions = struct();
    trainOptions.maxEpochs       = 60;
    trainOptions.miniBatchSize   = 512;
    trainOptions.initialLearnRate = 0.001;
    trainOptions.focalGamma      = 2.0;
    trainOptions.focalAlpha      = 0.25;
    trainOptions.patience        = 10;
    trainOptions.lrPatience      = 5;
    trainOptions.lrFactor        = 0.5;
    trainOptions.outputDir       = outputDir;
    trainOptions.checkpointEvery = 1;    % Her epoch'ta checkpoint
    trainOptions.trainAUCEvery   = 5;    % Her 5 epoch'ta gerçek train AUC
    
    % GPU kontrolü
    if canUseGPU()
        gpuInfo = gpuDevice;
        fprintf('[GPU] %s bulundu (%.1f GB bellek)\n', gpuInfo.Name, gpuInfo.TotalMemory/1e9);
        trainOptions.useGPU = true;
    else
        fprintf('[CPU] GPU bulunamadı, CPU ile eğitim yapılacak.\n');
        fprintf('[İPUCU] Eğitim CPU''da yavaş olabilir. Batch size küçültmeyi deneyin.\n');
        trainOptions.useGPU = false;
        trainOptions.miniBatchSize = 256;  % CPU için daha küçük batch
    end
    
    % Eğitim (checkpoint'ten devam veya yeni)
    [trainedNet, history] = trainSepsisModel(net, data, trainOptions, resumeState);
    
    savePipelineState(pipelineStateFile, 4, 'Model Eğitimi', data, trainedNet, [], history);
end

%% ========================================================================
%  ADIM 5: DEĞERLENDİRME
%  ========================================================================

if startStep <= 5
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 5/7: DEĞERLENDİRME ━━━\n');
    results = evaluateSepsisModel(trainedNet, data, outputDir);
    
    savePipelineState(pipelineStateFile, 5, 'Değerlendirme', data, trainedNet, results, history);
end

%% ========================================================================
%  ADIM 6: GÖRSELLEŞTİRME
%  ========================================================================

if startStep <= 6
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 6/7: GÖRSELLEŞTİRME ━━━\n');
    plotResults(results, history, figureDir);
    
    savePipelineState(pipelineStateFile, 6, 'Görselleştirme', data, trainedNet, results, history);
end

%% ========================================================================
%  ADIM 7: RAPOR OLUŞTURMA
%  ========================================================================

if startStep <= 7
    fprintf('\n\n');
    fprintf('  ━━━ ADIM 7/7: RAPOR OLUŞTURMA ━━━\n');
    generateReport(results, history, outputDir);
end

%% ========================================================================
%  SON ÖZET
%  ========================================================================

fprintf('\n\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║                    PROJE TAMAMLANDI!                        ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  ROC-AUC:  %.4f                                           ║\n', results.rocAUC);
fprintf('║  PR-AUC:   %.4f                                           ║\n', results.prAUC);
fprintf('║  F1-Score: %.4f                                           ║\n', results.f1Score);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Çıktılar:                                                  ║\n');
fprintf('║    → results/best_model.mat      (Eğitilmiş model)         ║\n');
fprintf('║    → results/test_results.mat    (Test sonuçları)           ║\n');
fprintf('║    → results/training_history.mat(Eğitim geçmişi)          ║\n');
fprintf('║    → results/proje_raporu.txt    (Proje raporu)             ║\n');
fprintf('║    → results/figures/*.png       (Grafikler, 300 DPI)       ║\n');
fprintf('║    → results/checkpoints/        (Eğitim checkpoint''leri)   ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('\nBitiş: %s\n', datestr(now, 'dd.mm.yyyy HH:MM:SS'));

% Pipeline tamamlandı — state dosyasını temizle
if isfile(pipelineStateFile)
    delete(pipelineStateFile);
    fprintf('\n  [OK] Pipeline state dosyası temizlendi (tamamlandı).\n');
end


%% ========================================================================
%  YARDIMCI FONKSİYON: Pipeline State Kaydetme
%  ========================================================================

function savePipelineState(stateFile, stepNum, stepName, data, net, results, history)
% Pipeline ilerlemesini kaydeder — her adım sonunda çağrılır
    lastStep  = stepNum;
    timestamp = datestr(now, 'dd.mm.yyyy HH:MM:SS');

    if nargin < 5, net = []; end
    if nargin < 6, results = []; end
    if nargin < 7, history = []; end
    
    % Büyük veriyi kaydetmekten kaçın (data struct çok büyük olabilir)
    % Sadece meta bilgiyi ve modeli kaydet
    dataMeta = struct();
    if ~isempty(data)
        dataMeta.numFeatures = data.numFeatures;
        dataMeta.columnNames = data.columnNames;
        dataMeta.trainSize   = size(data.X_train, 3);
        dataMeta.valSize     = size(data.X_val, 3);
        dataMeta.testSize    = size(data.X_test, 3);
    end
    
    save(stateFile, 'lastStep', 'stepName', 'timestamp', 'dataMeta', '-v7.3');
    
    % Büyük nesneleri ayrı kaydet
    if ~isempty(net)
        save(stateFile, 'net', '-append');
    end
    if ~isempty(results)
        save(stateFile, 'results', '-append');
    end
    if ~isempty(history)
        save(stateFile, 'history', '-append');
    end
    
    fprintf('  [✓] Adım %d tamamlandı: %s\n', stepNum, stepName);
end
