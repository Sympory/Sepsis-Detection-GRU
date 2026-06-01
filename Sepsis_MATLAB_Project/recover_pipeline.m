% recover_pipeline.m
% Bu betik, eğitim tamamlandıktan sonra oluşan grafik kaydetme hatasını kurtarır
% ve pipeline'ı doğrudan Adım 5 (Değerlendirme) aşamasına geçirir.

fprintf('Pipeline durumu kurtarılıyor...\n');

projectRoot = fileparts(mfilename('fullpath'));
outputDir = fullfile(projectRoot, 'results');
pipelineStateFile = fullfile(outputDir, 'pipeline_state.mat');

% 1. En iyi modeli yükle
bestModelFile = fullfile(outputDir, 'best_model.mat');
if isfile(bestModelFile)
    fprintf('  → En iyi model yükleniyor: %s\n', bestModelFile);
    m = load(bestModelFile);
    net = m.net;
else
    error('En iyi model dosyası bulunamadı! Lütfen %s dosyasının varlığından emin olun.', bestModelFile);
end

% 2. Eğitim geçmişini son checkpoint'ten yükle
checkpointDir = fullfile(outputDir, 'checkpoints');
cpFiles = dir(fullfile(checkpointDir, 'checkpoint_epoch_*.mat'));
if ~isempty(cpFiles)
    % En son checkpoint'i bul
    [~, idx] = max([cpFiles.datenum]);
    lastCpFile = fullfile(cpFiles(idx).folder, cpFiles(idx).name);
    fprintf('  → Eğitim geçmişi yükleniyor: %s\n', lastCpFile);
    cp = load(lastCpFile, 'history');
    history = cp.history;
else
    fprintf('  → Checkpoint bulunamadı, boş geçmiş oluşturuluyor.\n');
    history = struct();
end

% 3. Veriyi yükle ve feature engineering uygula
fprintf('  → Veri yükleniyor ve işleniyor...\n');
dataPath = fullfile(projectRoot, 'data', 'sepsis_data.mat');
data = loadSepsisData(dataPath);
[data.X_train, newCols] = engineerFeatures(data.X_train, data.columnNames);
[data.X_val, ~]         = engineerFeatures(data.X_val, data.columnNames);
[data.X_test, ~]        = engineerFeatures(data.X_test, data.columnNames);
data.columnNames = newCols;
data.numFeatures = size(data.X_train, 1);

% 4. Pipeline state dosyasını oluştur (Adım 4 tamamlandı olarak)
lastStep  = 4;
stepName  = 'Model Eğitimi';
timestamp = datestr(now, 'dd.mm.yyyy HH:MM:SS');

dataMeta = struct();
dataMeta.numFeatures = data.numFeatures;
dataMeta.columnNames = data.columnNames;
dataMeta.trainSize   = size(data.X_train, 3);
dataMeta.valSize     = size(data.X_val, 3);
dataMeta.testSize    = size(data.X_test, 3);

save(pipelineStateFile, 'lastStep', 'stepName', 'timestamp', 'dataMeta', '-v7.3');
save(pipelineStateFile, 'net', 'history', '-append');

fprintf('\n[✓] Pipeline durumu başarıyla kurtarıldı!\n');
fprintf('Şimdi MATLAB Command Window''a "main" yazıp Enter''a basın ve\n');
fprintf('çıkan menüden "1" (Kaldığım yerden devam et) seçeneğini seçin.\n');
