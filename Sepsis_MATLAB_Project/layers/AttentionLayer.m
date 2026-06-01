classdef AttentionLayer < nnet.layer.Layer
    % AttentionLayer: Temporal (Zamana Bağlı) Dikkat Mekanizması
    % Bu katman, Bi-GRU'dan çıkan zaman serisi verilerinden hangi zaman
    % adımının Sepsis teşhisi için daha önemli olduğunu öğrenir.
    
    properties (Learnable)
        % Eğitilebilir Ağırlıklar (W) ve Sapma (Bias, b)
        W
        b
    end
    
    methods
        function layer = AttentionLayer(name, numFeatures, numTimeSteps)
            % Katman isimlendirmesi
            layer.Name = name;
            layer.Description = 'Temporal Attention Layer';
            
            % W (Ağırlık): Her bir özellik (feature) için bir ağırlık katsayısı
            % Başlangıçta rastgele küçük değerler atıyoruz.
            layer.W = randn([numFeatures, 1]) * 0.01; 
            
            % b (Bias): Her zaman adımı (örn. 6 saatlik pencere) için sapma değeri
            % Başlangıçta sıfır atıyoruz.
            layer.b = zeros([1, numTimeSteps]);
        end
        
        function Z = predict(layer, X)
            % X: Girdi verisi. MATLAB'da varsayılan olarak [Kanal/Özellik, Zaman, Batch]
            % boyutlarındadır. Yani [numFeatures, numTimeSteps, batchSize].
            
            % 1. Adım: Her zaman adımı için e değerini hesapla
            % X ile W matrisini çarpıyoruz.
            e = sum(X .* layer.W, 1); % Boyut: [1, numTimeSteps, batchSize]
            
            % Bias ekle
            e = e + layer.b;
            
            % tanh aktivasyon fonksiyonundan geçir
            e = tanh(e);
            
            % 2. Adım: Zaman ekseni boyunca (Boyut 2) Softmax uygula (Dikkat Ağırlıkları)
            % exp(e) değerlerini, zaman eksenindeki toplam exp(e) değerine bölüyoruz.
            exp_e = exp(e);
            a = exp_e ./ sum(exp_e, 2); % 'a' artık hangi saatin daha önemli olduğunu (0-1 arası) gösterir
            
            % 3. Adım: Orijinal girdi (X) ile dikkat ağırlıklarını (a) çarp
            output = X .* a;
            
            % 4. Adım: Tüm zaman adımlarını toplayarak tek bir bağlam vektörü (Context Vector) oluştur
            Z = sum(output, 2); % Boyut: [numFeatures, 1, batchSize]
            
            % Boyutu [numFeatures, batchSize] olarak düzelt (Gereksiz 1 boyutunu at)
            Z = squeeze(Z);
        end
    end
end
