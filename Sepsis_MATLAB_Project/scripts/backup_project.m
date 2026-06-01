function backupPath = backup_project(projectRoot, backupDir)
% BACKUP_PROJECT Proje dosyalarının tarih damgalı zip yedeğini oluşturur
%   backupPath = backup_project(projectRoot)
%   backupPath = backup_project(projectRoot, backupDir)
%
%   Eğitim öncesinde veya herhangi bir değişiklik öncesinde çalıştırarak
%   projenin tam yedeğini alır. Böylece herhangi bir sorun durumunda
%   eski haline dönebilirsiniz.
%
%   Girdi:
%     projectRoot - Proje kök dizini (varsayılan: Sepsis_MATLAB_Project/)
%     backupDir   - Yedek dosyaların saklanacağı klasör
%                   (varsayılan: projectRoot/../backups/)
%
%   Çıktı:
%     backupPath  - Oluşturulan zip dosyasının tam yolu
%
%   Kullanım:
%     >> backup_project()                        % Varsayılan konuma yedekle
%     >> backup_project(pwd, 'D:\yedeklerim')    % Özel konuma yedekle

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('PROJE YEDEKLEMESİ\n');
    fprintf('%s\n', repmat('=', 1, 60));

    % Varsayılan proje kökü
    if nargin < 1 || isempty(projectRoot)
        projectRoot = fileparts(fileparts(mfilename('fullpath')));
    end

    % Varsayılan yedek dizini
    if nargin < 2 || isempty(backupDir)
        backupDir = fullfile(fileparts(projectRoot), 'backups');
    end

    % Yedek dizinini oluştur
    if ~isfolder(backupDir)
        mkdir(backupDir);
        fprintf('  [+] Yedek klasörü oluşturuldu: %s\n', backupDir);
    end

    % Zaman damgası
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    zipName = sprintf('Sepsis_MATLAB_backup_%s.zip', timestamp);
    backupPath = fullfile(backupDir, zipName);

    fprintf('  Kaynak:  %s\n', projectRoot);
    fprintf('  Hedef:   %s\n', backupPath);

    % Yedeklenecek dosyaları topla (büyük veri dosyalarını hariç tut)
    fprintf('\n  Dosyalar taranıyor...\n');

    allFiles = {};
    excludePatterns = {'*.mat', '*.npy', '*.pkl', '*.zip', '.git'};

    % .m dosyaları (en önemli)
    mFiles = dir(fullfile(projectRoot, '**', '*.m'));
    for i = 1:numel(mFiles)
        allFiles{end+1} = fullfile(mFiles(i).folder, mFiles(i).name); %#ok<AGROW>
    end

    % .py dosyaları
    pyFiles = dir(fullfile(projectRoot, '**', '*.py'));
    for i = 1:numel(pyFiles)
        allFiles{end+1} = fullfile(pyFiles(i).folder, pyFiles(i).name); %#ok<AGROW>
    end

    % Küçük boyutlu diğer dosyalar (< 10 MB)
    otherFiles = dir(fullfile(projectRoot, '**', '*.*'));
    for i = 1:numel(otherFiles)
        if otherFiles(i).isdir, continue; end
        fpath = fullfile(otherFiles(i).folder, otherFiles(i).name);

        % Zaten eklenmişse atla
        if any(strcmp(allFiles, fpath)), continue; end

        % Büyük dosyaları ve hariç tutulanları atla
        if otherFiles(i).bytes > 10 * 1024 * 1024, continue; end

        skip = false;
        for j = 1:numel(excludePatterns)
            if contains(fpath, strrep(excludePatterns{j}, '*', ''))
                skip = true;
                break;
            end
        end
        if ~skip
            allFiles{end+1} = fpath; %#ok<AGROW>
        end
    end

    % Benzersiz yap
    allFiles = unique(allFiles);

    fprintf('  %d dosya yedeklenecek\n', numel(allFiles));

    % Zip oluştur
    try
        zip(backupPath, allFiles);
        fileInfo = dir(backupPath);
        fprintf('\n  [OK] Yedek oluşturuldu!\n');
        fprintf('  Dosya: %s\n', backupPath);
        fprintf('  Boyut: %.1f KB\n', fileInfo.bytes / 1024);
    catch ME
        fprintf('\n  [HATA] Yedekleme başarısız: %s\n', ME.message);
        backupPath = '';
        return;
    end

    % Eski yedekleri listele
    existingBackups = dir(fullfile(backupDir, 'Sepsis_MATLAB_backup_*.zip'));
    if numel(existingBackups) > 1
        fprintf('\n  Mevcut yedekler (%d adet):\n', numel(existingBackups));
        for i = 1:numel(existingBackups)
            fprintf('    %d. %s (%.1f KB)\n', i, existingBackups(i).name, ...
                existingBackups(i).bytes / 1024);
        end

        % 10'dan fazla yedek varsa uyar
        if numel(existingBackups) > 10
            fprintf('\n  [UYARI] 10''dan fazla yedek var. Eski yedekleri silmeyi düşünün.\n');
        end
    end

    fprintf('\n%s\n', repmat('=', 1, 60));
end
