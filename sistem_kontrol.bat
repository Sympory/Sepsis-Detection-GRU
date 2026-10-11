@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PY="
py -3.10 -c "import sys" >nul 2>&1 && set "PY=py -3.10"
if not defined PY set "PY=python"
%PY% sistem_kontrol.py
pause
