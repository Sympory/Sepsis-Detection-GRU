function [net, state] = restoreCheckpoint(checkpointDir, epochNum)
% RESTORECHECKPOINT Checkpoint'ten model ve eğitim durumunu geri yükler
%   [net, state] = restoreCheckpoint(checkpointDir)
%   [net, state] = restoreCheckpoint(checkpointDir, epochNum)
%
%   Eğitim sırasında kaydedilen checkpoint dosyalarını yönetir.
%   Epoch numarası verilmezse mevcut checkpoint'leri listeler ve
%   kullanıcıdan seçim yapmasını ister.
%
%   Girdi:
%     checkpointDir - Checkpoint dosyalarının bulunduğu klasör
%                     (varsayılan: 'results/checkpoints/')
%     epochNum      - Yüklenecek epoch numarası (opsiyonel)
%
%   Çıktı:
%     net   - Kaydedilmiş dlnetwork modeli
%     state - Eğitim durumu struct'ı:
%       .epoch          - Son tamamlanan epoch
%       .avgGrad        - Adam optimizer birinci moment
%       .avgSqGrad      - Adam optimizer ikinci moment
%       .iteration      - Toplam iterasyon sayısı
%       .learnRate      - Son learning rate
%       .history        - Eğitim geçmişi
%       .bestNet        - En iyi model (val PR-AUC'ye göre)
%       .bestValPR_AUC  - En iyi validation PR-AUC
%       .bestEpoch      - En iyi epoch numarası
%
%   Kullanım:
%     % Checkpoint'leri listele ve seç
%     >> [net, state] = restoreCheckpoint('results/checkpoints')
%
%     % Belirli epoch'u yükle
%     >> [net, state] = restoreCheckpoint('results/checkpoints', 15)
%
%     % Eğitime devam et
%     >> [net, state] = restoreCheckpoint('results/checkpoints', 15);
%     >> % state'i trainSepsisModel'e aktar

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('CHECKPOINT YÖNETİCİSİ\n');
    fprintf('%s\n', repmat('=', 1, 60));

    % Varsayılan dizin
    if nargin < 1 || isempty(checkpointDir)
        projectRoot = fileparts(fileparts(mfilename('fullpath')));
        checkpointDir = fullfile(projectRoot, 'results', 'checkpoints');
    end

    % Dizin kontrolü
    if ~isfolder(checkpointDir)
        fprintf('\n  [HATA] Checkpoint klasörü bulunamadı: %s\n', checkpointDir);
        fprintf('  [BİLGİ] Henüz eğitim yapılmamış olabilir.\n');
        net = [];
        state = struct();
        return;
    end

    % Mevcut checkpoint'leri bul
    cpFiles = dir(fullfile(checkpointDir, 'checkpoint_epoch_*.mat'));

    if isempty(cpFiles)
        fprintf('\n  [HATA] Hiç checkpoint dosyası bulunamadı!\n');
        fprintf('  Klasör: %s\n', checkpointDir);
        net = [];
        state = struct();
        return;
    end

    % Epoch numaralarını çıkar ve sırala
    epochs = zeros(numel(cpFiles), 1);
    for i = 1:numel(cpFiles)
        tokens = regexp(cpFiles(i).name, 'checkpoint_epoch_(\d+)', 'tokens');
        if ~isempty(tokens)
            epochs(i) = str2double(tokens{1}{1});
        end
    end
    [epochs, sortIdx] = sort(epochs);
    cpFiles = cpFiles(sortIdx);

    % Checkpoint listesini göster
    fprintf('\n  Mevcut Checkpoint''ler:\n');
    fprintf('  %s\n', repmat('-', 1, 55));
    fprintf('  %-6s | %-20s | %-10s | %-12s\n', 'Epoch', 'Tarih', 'Boyut', 'Val PR-AUC');
    fprintf('  %s\n', repmat('-', 1, 55));

    for i = 1:numel(cpFiles)
        fInfo = cpFiles(i);
        sizeKB = fInfo.bytes / 1024;
        dateStr = datestr(fInfo.datenum, 'dd.mm.yyyy HH:MM');

        % PR-AUC bilgisini hızlıca oku (varsa)
        try
            cpData = load(fullfile(fInfo.folder, fInfo.name), 'bestValPR_AUC');
            prAUC = sprintf('%.4f', cpData.bestValPR_AUC);
        catch
            prAUC = '—';
        end

        fprintf('  %-6d | %-20s | %7.1f KB | %-12s\n', ...
            epochs(i), dateStr, sizeKB, prAUC);
    end
    fprintf('  %s\n', repmat('-', 1, 55));

    % Epoch seçimi
    if nargin < 2 || isempty(epochNum)
        fprintf('\n  Hangi epoch''u yüklemek istiyorsunuz?\n');
        fprintf('  (En son = %d, Enter = en son): ', epochs(end));
        userInput = input('', 's');

        if isempty(userInput)
            epochNum = epochs(end);
        else
            epochNum = str2double(userInput);
        end
    end

    % Geçerlilik kontrolü
    if ~ismember(epochNum, epochs)
        fprintf('\n  [HATA] Epoch %d bulunamadı!\n', epochNum);
        fprintf('  Mevcut epoch''lar: %s\n', mat2str(epochs'));
        net = [];
        state = struct();
        return;
    end

    % Checkpoint'i yükle
    cpFile = fullfile(checkpointDir, sprintf('checkpoint_epoch_%d.mat', epochNum));
    fprintf('\n  Yükleniyor: %s\n', cpFile);

    try
        cpData = load(cpFile);
    catch ME
        fprintf('  [HATA] Dosya yüklenemedi: %s\n', ME.message);
        net = [];
        state = struct();
        return;
    end

    % Model ve durumu çıkar
    net = cpData.net;

    state = struct();
    state.epoch         = cpData.epoch;
    state.avgGrad       = cpData.avgGrad;
    state.avgSqGrad     = cpData.avgSqGrad;
    state.iteration     = cpData.iteration;
    state.learnRate     = cpData.learnRate;
    state.history       = cpData.history;
    state.bestNet       = cpData.bestNet;
    state.bestValPR_AUC = cpData.bestValPR_AUC;
    state.bestEpoch     = cpData.bestEpoch;

    fprintf('\n  [OK] Checkpoint başarıyla yüklendi!\n');
    fprintf('  ┌─────────────────────────────────────────┐\n');
    fprintf('  │  Epoch:          %-24d│\n', state.epoch);
    fprintf('  │  Learning Rate:  %-24.6f│\n', state.learnRate);
    fprintf('  │  Best PR-AUC:    %-24.4f│\n', state.bestValPR_AUC);
    fprintf('  │  Best Epoch:     %-24d│\n', state.bestEpoch);
    fprintf('  │  İterasyon:      %-24d│\n', state.iteration);
    fprintf('  └─────────────────────────────────────────┘\n');

    fprintf('\n  Eğitime devam etmek için:\n');
    fprintf('    >> [trainedNet, history] = trainSepsisModel(net, data, options, state);\n');

    fprintf('\n%s\n', repmat('=', 1, 60));
end
