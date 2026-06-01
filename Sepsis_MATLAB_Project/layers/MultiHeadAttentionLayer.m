classdef MultiHeadAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    % MultiHeadAttentionLayer: Çok-Başlı Temporal Dikkat Mekanizması
    %
    %   Bi-LSTM çıktısından hangi zaman adımlarının Sepsis teşhisi
    %   için daha önemli olduğunu öğrenir.
    %
    %   Girdi:  [features, timeSteps, batch] (CTB)
    %   Çıktı:  [units, 1, batch]            (CTB, T=1)
    
    properties
        NumHeads    (1,1) double = 4
        Units       (1,1) double = 128
    end
    
    properties (Learnable)
        QueryWeights
        KeyWeights
        ValueWeights
        OutputWeights
        OutputBias
    end
    
    methods
        function layer = MultiHeadAttentionLayer(numInputFeatures, units, numHeads, name)
            arguments
                numInputFeatures (1,1) double
                units            (1,1) double = 128
                numHeads         (1,1) double = 4
                name             string       = "multi_head_attention"
            end
            
            layer.Name = name;
            layer.Units = units;
            layer.NumHeads = numHeads;
            layer.Description = sprintf('Multi-Head Temporal Attention (%d heads, %d units)', ...
                numHeads, units);
            
            F = numInputFeatures;
            U = units;
            scale = sqrt(2 / (F + U));
            
            layer.QueryWeights  = randn(F, U, 'single') * scale;
            layer.KeyWeights    = randn(F, U, 'single') * scale;
            layer.ValueWeights  = randn(F, U, 'single') * scale;
            layer.OutputWeights = randn(U, U, 'single') * scale;
            layer.OutputBias    = zeros(U, 1, 'single');
        end
        
        function Z = predict(layer, X)
            % X: dlarray [features, timeSteps, batch] (CTB formatı)
            % Z: dlarray [units, 1, batch]            (CTB formatı, T=1)
            
            % Format etiketlerini kaldır (pagemtimes/permute uyumu için)
            X_raw = stripdims(X);
            
            % ---- 1. Q, K, V Projeksiyon ----
            Q = pagemtimes(layer.QueryWeights', X_raw);   % [U, T, B]
            K = pagemtimes(layer.KeyWeights', X_raw);
            V = pagemtimes(layer.ValueWeights', X_raw);
            
            % ---- 2. Scaled Dot-Product Attention ----
            d_k = layer.Units / layer.NumHeads;
            scores = pagemtimes(permute(Q, [2,1,3]), K) / sqrt(d_k); % [T, T, B]
            
            % Numerik kararlı Softmax
            scores_max = max(scores, [], 2);
            scores_exp = exp(scores - scores_max);
            weights = scores_exp ./ sum(scores_exp, 2);
            
            % context = weights * V^T → [T, U, B]
            context = pagemtimes(weights, permute(V, [2,1,3]));
            
            % ---- 3. Global Average Pooling (Zaman boyutu) ----
            context_avg = mean(context, 1);              % [1, U, B]
            context_avg = permute(context_avg, [2,1,3]); % [U, 1, B]
            
            % ---- 4. Çıkış Projeksiyon ----
            Z_raw = pagemtimes(layer.OutputWeights', context_avg) + layer.OutputBias;
            % Z_raw: [U, 1, B]
            
            % Format etiketlerini geri ekle — sonraki katmanlar doğru çalışsın
            Z = dlarray(Z_raw, 'CTB');
        end
    end
end
