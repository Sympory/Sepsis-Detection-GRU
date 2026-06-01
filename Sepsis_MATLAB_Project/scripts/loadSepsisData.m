function data = loadSepsisData(matFilePath)
% LOADSEPSISDATA Sepsis veri setini MATLAB formatında yükler
%   data = loadSepsisData(matFilePath)
%
%   Girdi:
%     matFilePath - .mat dosyasının tam yolu
%
%   Çıktı:
%     data - Struct: X_train, y_train, X_val, y_val, X_test, y_test,
%                    columnNames, windowSize, numFeatures, classWeights
%
%   Veri boyutları (MATLAB Deep Learning formatı):
%     X: [numFeatures, windowSize, numSamples] (CTB formatı)
%     y: [1, numSamples]

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('VERİ YÜKLEME\n');
    fprintf('%s\n', repmat('=', 1, 60));

    if nargin < 1
        % Varsayılan yol
        scriptDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(scriptDir);
        matFilePath = fullfile(projectRoot, 'data', 'sepsis_data.mat');
    end

    fprintf('Dosya: %s\n', matFilePath);
    
    if ~isfile(matFilePath)
        error('Veri dosyası bulunamadı: %s\nÖnce convert_npy_to_mat.py çalıştırın.', matFilePath);
    end

    % --- Veriyi Yükle ---
    fprintf('\n[1/3] MAT dosyası yükleniyor...\n');
    raw = load(matFilePath);
    
    % --- Boyut Dönüşümü ---
    % Python formatı: [N, T, F] → MATLAB Deep Learning: [F, T, N] (CTB)
    fprintf('[2/3] Boyut dönüşümü yapılıyor (NTF → CTB)...\n');
    
    data.X_train = single(permute(raw.X_train, [3, 2, 1]));  % [F, T, N]
    data.y_train = single(raw.y_train');                       % [1, N]
    
    data.X_val = single(permute(raw.X_val, [3, 2, 1]));
    data.y_val = single(raw.y_val');
    
    data.X_test = single(permute(raw.X_test, [3, 2, 1]));
    data.y_test = single(raw.y_test');
    
    % Meta bilgi
    data.columnNames = cellstr(raw.column_names);
    data.windowSize = double(raw.window_size);
    data.numFeatures = double(raw.num_features);
    
    % --- Sınıf Ağırlıkları ---
    fprintf('[3/3] Sınıf ağırlıkları hesaplanıyor...\n');
    nTotal = numel(data.y_train);
    nPos = sum(data.y_train == 1);
    nNeg = nTotal - nPos;
    
    data.classWeights.negative = nTotal / (2 * nNeg);
    data.classWeights.positive = nTotal / (2 * nPos);
    data.classWeights.ratio = data.classWeights.positive / data.classWeights.negative;
    
    % --- Özet ---
    fprintf('\n--- Veri Özeti ---\n');
    fprintf('  Train:  %d örnek (%.2f%% Sepsis)\n', ...
        size(data.X_train, 3), (nPos/nTotal)*100);
    fprintf('  Val:    %d örnek (%.2f%% Sepsis)\n', ...
        size(data.X_val, 3), mean(data.y_val)*100);
    fprintf('  Test:   %d örnek (%.2f%% Sepsis)\n', ...
        size(data.X_test, 3), mean(data.y_test)*100);
    fprintf('  Özellik sayısı: %d\n', data.numFeatures);
    fprintf('  Pencere boyutu: %d saat\n', data.windowSize);
    fprintf('  Sınıf ağırlığı oranı: 1:%.1f\n', data.classWeights.ratio);
    fprintf('%s\n', repmat('=', 1, 60));
end
