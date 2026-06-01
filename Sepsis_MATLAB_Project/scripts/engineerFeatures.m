function [X_enhanced, newColumnNames] = engineerFeatures(X, columnNames)
% ENGINEERFEATURES 12 yeni temporal ve etkileşim özelliği oluşturur
%   [X_enhanced, newColumnNames] = engineerFeatures(X, columnNames)
%
%   Python'daki engineer_features() fonksiyonunun MATLAB karşılığı.
%
%   Girdi:
%     X           - [numFeatures, numTimeSteps, numSamples] (CTB formatı)
%     columnNames - cell array, özellik isimleri
%
%   Çıktı:
%     X_enhanced     - [numFeatures+12, numTimeSteps, numSamples]
%     newColumnNames - güncellenmiş özellik isimleri
%
%   Eklenen 12 Yeni Özellik:
%     Delta (Değişim Hızı):
%       1. dHR       - Kalp atış hızı değişimi
%       2. dTemp     - Vücut sıcaklığı değişimi
%       3. dMAP      - Ortalama arter basıncı değişimi
%       4. dLactate  - Laktat değişimi
%       5. dResp     - Solunum hızı değişimi
%       6. dO2Sat    - Oksijen satürasyonu değişimi
%     Etkileşim (Interaction):
%       7. ShockIndex   - HR / SBP (Şok İndeksi)
%       8. MAP_HR_Ratio - MAP / HR
%       9. O2_Delivery  - MAP * O2Sat / 100
%       10. Resp_HR     - Resp / HR
%     Değişkenlik (Variability):
%       11. HR_var   - HR rolling standart sapması
%       12. Temp_var - Sıcaklık rolling standart sapması

    fprintf('\n--- Feature Engineering (12 Yeni Özellik) ---\n');
    
    [F, T, N] = size(X);
    fprintf('  Girdi boyutu: [%d, %d, %d]\n', F, T, N);
    
    % Sütun indeks haritası oluştur
    colIdx = containers.Map();
    for i = 1:length(columnNames)
        colIdx(columnNames{i}) = i;
    end
    
    % Yeni özellikler için yer ayır: [12, T, N]
    newFeatures = zeros(12, T, N, 'single');
    
    % ================================================================
    % 1. DELTA ÖZELLİKLERİ (Değişim Hızı: x(t) - x(t-1))
    % ================================================================
    deltaNames = {'HR', 'Temp', 'MAP', 'Lactate', 'Resp', 'O2Sat'};
    
    for i = 1:length(deltaNames)
        name = deltaNames{i};
        if colIdx.isKey(name)
            idx = colIdx(name);
            vals = squeeze(X(idx, :, :));    % [T, N]
            delta = zeros(size(vals), 'single');
            delta(2:end, :) = vals(2:end, :) - vals(1:end-1, :);
            % delta(1,:) = 0 (ilk zaman adımında delta yok)
            newFeatures(i, :, :) = reshape(delta, [1, T, N]);
        end
    end
    fprintf('  [OK] 6 Delta özelliği hesaplandı\n');
    
    % ================================================================
    % 2. ETKİLEŞİM ÖZELLİKLERİ
    % ================================================================
    
    % Güvenli indeks alma fonksiyonu
    getCol = @(name, default) getColumnSafe(X, colIdx, name, default);
    
    hr   = getCol('HR', 1);     % [T, N]
    sbp  = getCol('SBP', 4);
    mapV = getCol('MAP', 5);
    o2   = getCol('O2Sat', 2);
    resp = getCol('Resp', 7);
    
    % Sıfıra bölme koruması
    hr_safe  = max(abs(hr), 1e-6) .* sign(hr + 1e-10);
    sbp_safe = max(abs(sbp), 1e-6) .* sign(sbp + 1e-10);
    
    % ShockIndex = HR / SBP
    shockIdx = hr ./ sbp_safe;
    newFeatures(7, :, :) = reshape(shockIdx, [1, T, N]);
    
    % MAP / HR Oranı
    mapHrRatio = mapV ./ hr_safe;
    newFeatures(8, :, :) = reshape(mapHrRatio, [1, T, N]);
    
    % O2 Delivery Proxy = MAP * O2Sat / 100
    o2Delivery = mapV .* o2;
    newFeatures(9, :, :) = reshape(o2Delivery, [1, T, N]);
    
    % Resp / HR Oranı
    respHr = resp ./ hr_safe;
    newFeatures(10, :, :) = reshape(respHr, [1, T, N]);
    
    fprintf('  [OK] 4 Etkileşim özelliği hesaplandı\n');
    
    % ================================================================
    % 3. DEĞİŞKENLİK ÖZELLİKLERİ (Rolling Std)
    % ================================================================
    tempVals = getCol('Temp', 3);
    
    hrVar   = zeros(T, N, 'single');
    tempVar = zeros(T, N, 'single');
    
    for t = 1:T
        % t anına kadar olan pencerenin standart sapması
        window_hr = hr(1:t, :);       % [t, N]
        window_temp = tempVals(1:t, :);
        
        if t == 1
            hrVar(t, :)   = 0;
            tempVar(t, :) = 0;
        else
            hrVar(t, :)   = std(window_hr, 0, 1);    % Her hasta için std
            tempVar(t, :) = std(window_temp, 0, 1);
        end
    end
    
    newFeatures(11, :, :) = reshape(hrVar, [1, T, N]);
    newFeatures(12, :, :) = reshape(tempVar, [1, T, N]);
    
    fprintf('  [OK] 2 Değişkenlik özelliği hesaplandı\n');
    
    % ================================================================
    % 4. TEMİZLİK & BİRLEŞTİRME
    % ================================================================
    
    % NaN ve Inf temizliği
    newFeatures(isnan(newFeatures)) = 0;
    newFeatures(isinf(newFeatures) & newFeatures > 0) = 3.0;
    newFeatures(isinf(newFeatures) & newFeatures < 0) = -3.0;
    
    % Aşırı değerleri kırp (z-score normalize edilmiş veri üzerinde)
    newFeatures = max(min(newFeatures, 5.0), -5.0);
    
    % Birleştir: [F+12, T, N]
    X_enhanced = cat(1, X, newFeatures);
    
    % Yeni sütun isimleri
    engineeredNames = {'dHR', 'dTemp', 'dMAP', 'dLactate', 'dResp', 'dO2Sat', ...
                       'ShockIndex', 'MAP_HR_Ratio', 'O2_Delivery', 'Resp_HR', ...
                       'HR_var', 'Temp_var'};
    newColumnNames = [columnNames(:)', engineeredNames];
    
    fprintf('  [OK] Toplam özellik: %d → %d\n', F, size(X_enhanced, 1));
    fprintf('--- Feature Engineering Tamamlandı ---\n');
end


function vals = getColumnSafe(X, colIdx, name, defaultIdx)
% Sütun adına göre veriyi döndürür, bulunamazsa varsayılan indeks kullanır
    if colIdx.isKey(name)
        idx = colIdx(name);
    else
        idx = defaultIdx;
        fprintf('  [UYARI] "%s" sütunu bulunamadı, indeks %d kullanılıyor\n', name, idx);
    end
    % X: [F, T, N] → vals: [T, N]
    vals = squeeze(X(idx, :, :));
end
