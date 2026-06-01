% Sepsis_MATLAB_Project Başlangıç Betiği (Startup Script)
% Bu dosya projedeki tüm alt klasörleri MATLAB yoluna (path) ekler.
% Böylece klasörler arası dosya arama karmaşası önlenir.

disp('Sepsis_MATLAB_Project Yükleniyor...');

% Projenin ana klasör yolunu al
projectRoot = fileparts(mfilename('fullpath'));

% Alt klasörleri yola ekle
addpath(fullfile(projectRoot, 'layers'));
addpath(fullfile(projectRoot, 'losses'));
addpath(fullfile(projectRoot, 'scripts'));
addpath(fullfile(projectRoot, 'data'));
addpath(fullfile(projectRoot, 'results'));

disp('Proje yolları (path) başarıyla eklendi.');
disp('  → layers/   (Custom katmanlar: MultiHeadAttentionLayer, AttentionLayer)');
disp('  → losses/   (Kayıp fonksiyonları: focalLoss)');
disp('  → scripts/  (Ana fonksiyonlar: load, build, train, evaluate, plot)');
disp('  → data/     (Veri dosyaları)');
disp('  → results/  (Sonuçlar ve grafikler)');
disp('Kullanıma Hazır!');
