@echo off
REM ============================================================
REM  SepsisAI - tek tikla baslatma
REM  Python 3.10 bulur, eksik paketleri kurar, uygulamayi acar.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title SepsisAI - YBU Karar Destek

set "PY="
py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [HATA] Python 3.10 bulunamadi. TensorFlow 2.10 icin Python 3.10 gerekli.
    echo        https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)
echo [OK] Python: %PY%

%PY% -c "import tensorflow, flask, flask_cors, sklearn, pandas" >nul 2>&1
if errorlevel 1 (
    echo [..] Eksik paketler kuruluyor, bu birkac dakika surebilir...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [HATA] Paket kurulumu basarisiz.
        pause
        exit /b 1
    )
)

set "SEPSIS_OPEN_BROWSER=1"
set "PYTHONIOENCODING=utf-8"
echo [..] Model yukleniyor, tarayici otomatik acilacak (yaklasik 20 sn)...
%PY% app.py
echo.
echo Uygulama kapandi.
pause
