function net = buildEnhancedModel(numFeatures, numTimeSteps)
% BUILDENHANCEDMODEL Gelişmiş Sepsis tahmin modelini oluşturur (Enhanced v2)
%   net = buildEnhancedModel(numFeatures, numTimeSteps)
%
%   Mimari:
%     Input(F, T) → Bi-LSTM(96) → LayerNorm → Dropout(0.3)
%                 → Bi-LSTM(64) → LayerNorm → Dropout(0.3)
%                 → Self-Attention(4 heads, 128 key)
%                 → Global Average Pooling (zaman boyutunu daralt)
%                 → Dense(64, ReLU) → BatchNorm → Dropout(0.3)
%                 → Dense(32, ReLU) → BatchNorm → Dropout(0.2)
%                 → Dense(1, Sigmoid)
%
%   Not: MATLAB'ın built-in selfAttentionLayer ve 
%   globalAveragePooling1dLayer katmanları kullanılır.
%   Custom layer gerekmez, format uyumu otomatiktir.

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('MODEL MİMARİSİ (Enhanced v2)\n');
    fprintf('%s\n', repmat('=', 1, 60));

    % ================================================================
    % KATMAN TANIMLAMALARI (Tümü Built-in)
    % ================================================================
    
    layers = [
        % ---- Giriş ----
        sequenceInputLayer(numFeatures, ...
            'Name', 'input', ...
            'Normalization', 'none', ...
            'MinLength', numTimeSteps)
        
        % ---- Bi-LSTM Katman 1 (96 ünite) ----
        % Çift yönlü: ileri + geri = 2*96 = 192 çıktı
        bilstmLayer(96, ...
            'Name', 'bi_lstm_1', ...
            'OutputMode', 'sequence')
        
        layerNormalizationLayer('Name', 'ln_1')
        dropoutLayer(0.3, 'Name', 'drop_1')
        
        % ---- Bi-LSTM Katman 2 (64 ünite) ----
        % 2*64 = 128 çıktı
        bilstmLayer(64, ...
            'Name', 'bi_lstm_2', ...
            'OutputMode', 'sequence')
        
        layerNormalizationLayer('Name', 'ln_2')
        dropoutLayer(0.3, 'Name', 'drop_2')
        
        % ---- Self-Attention (Built-in) ----
        % 4 dikkat başı, 128 anahtar boyutu
        % Her zaman adımını diğerleriyle karşılaştırarak
        % hangi saatlerin kritik olduğunu öğrenir
        selfAttentionLayer(4, 128, 'Name', 'self_attention')
        
        % ---- Global Average Pooling ----
        % Zaman boyutunu daraltır: [C, T, B] → [C, 1, B] → [C, B]
        % Tüm zaman adımlarının ağırlıklı ortalaması
        globalAveragePooling1dLayer('Name', 'gap')
        
        % ---- Sınıflandırıcı Blok 1 ----
        fullyConnectedLayer(64, 'Name', 'dense_1')
        reluLayer('Name', 'relu_1')
        batchNormalizationLayer('Name', 'bn_1')
        dropoutLayer(0.3, 'Name', 'drop_3')
        
        % ---- Sınıflandırıcı Blok 2 ----
        fullyConnectedLayer(32, 'Name', 'dense_2')
        reluLayer('Name', 'relu_2')
        batchNormalizationLayer('Name', 'bn_2')
        dropoutLayer(0.2, 'Name', 'drop_4')
        
        % ---- Çıkış ----
        fullyConnectedLayer(1, 'Name', 'output_logits')
        sigmoidLayer('Name', 'output')
    ];
    
    % ================================================================
    % dlnetwork OLUŞTUR
    % ================================================================
    
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
    
    % ================================================================
    % MODEL ÖZETİ
    % ================================================================
    
    fprintf('\n--- Model Özeti ---\n');
    fprintf('  Mimari: Bi-LSTM(96) → Bi-LSTM(64) → SelfAttention(4h,128)\n');
    fprintf('          → GAP → Dense(64) → Dense(32) → Dense(1,Sigmoid)\n');
    fprintf('  Giriş:  [%d özellik, %d zaman adımı]\n', numFeatures, numTimeSteps);
    totalParams = sum(cellfun(@numel, net.Learnables.Value));
    fprintf('  Toplam parametre: %s\n', formatNumber(totalParams));
    fprintf('%s\n', repmat('=', 1, 60));
    
    % Katman detaylarını göster
    fprintf('\n  %-30s %-40s\n', 'Katman', 'Tip');
    fprintf('  %s\n', repmat('-', 1, 70));
    for i = 1:numel(net.Layers)
        fprintf('  %-30s %-40s\n', net.Layers(i).Name, class(net.Layers(i)));
    end
    fprintf('\n');
end


function str = formatNumber(n)
    n = abs(n);
    if n >= 1e6
        str = sprintf('%.2fM', n/1e6);
    elseif n >= 1e3
        str = sprintf('%.1fK', n/1e3);
    else
        str = sprintf('%d', round(n));
    end
end
