function loss = focalLoss(YPred, YTrue, gamma, alpha)
% FOCALLOSS Focal Loss - Dengesiz Sınıflar İçin Kayıp Fonksiyonu
%   loss = focalLoss(YPred, YTrue, gamma, alpha)
%
%   Focal Loss, standart cross-entropy'nin geliştirilmiş halidir.
%   Kolay sınıflandırılan örneklerin ağırlığını azaltır,
%   zor örneklere odaklanmayı sağlar. Sepsis tespiti gibi
%   dengesiz veri setlerinde (pozitif << negatif) çok etkilidir.
%
%   Formül:
%     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
%
%   Girdi:
%     YPred - Model tahminleri (sigmoid sonrası, 0-1 arası), dlarray
%     YTrue - Gerçek etiketler (0 veya 1), dlarray
%     gamma - Odaklanma parametresi (varsayılan: 2.0)
%             gamma↑ → kolay örneklerin ağırlığı↓
%     alpha - Pozitif sınıf ağırlığı (varsayılan: 0.25)
%
%   Çıktı:
%     loss  - Skaler kayıp değeri (dlarray uyumlu, gradient hesaplanabilir)
%
%   Referans:
%     Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    if nargin < 3
        gamma = 2.0;
    end
    if nargin < 4
        alpha = 0.25;
    end

    % Numerik kararlılık için tahminleri sınırla
    epsilon = 1e-7;
    YPred = max(YPred, epsilon);
    YPred = min(YPred, 1 - epsilon);

    % p_t hesapla: doğru sınıfın olasılığı
    %   y=1 ise p_t = p
    %   y=0 ise p_t = 1-p
    pt = YTrue .* YPred + (1 - YTrue) .* (1 - YPred);

    % alpha_t: sınıf ağırlığı
    %   y=1 ise alpha_t = alpha
    %   y=0 ise alpha_t = 1-alpha
    alpha_t = YTrue .* alpha + (1 - YTrue) .* (1 - alpha);

    % Cross-entropy terimi
    ce = -log(pt);

    % Focal ağırlık: kolay örnekleri baskıla
    focalWeight = alpha_t .* ((1 - pt) .^ gamma);

    % Toplam kayıp (batch ortalaması)
    loss = mean(focalWeight .* ce, 'all');
end
