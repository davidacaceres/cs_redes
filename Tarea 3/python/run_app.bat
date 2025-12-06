@echo off
title Tarea 3 - Analisis de Redes
echo ==========================================
echo    INICIANDO APLICACION TAREA 3
echo ==========================================
echo.

if not exist ".venv" (
    echo [ERROR] No se detecto el entorno virtual '.venv'.
    echo Por favor ejecute 'install.bat' primero.
    pause
    exit /b 1
)

echo [INFO] Activando entorno y ejecutando aplicacion...
call .venv\Scripts\activate.bat

python tarea-3.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] La aplicacion se cerro con errores.
    pause
) else (
    echo.
    echo [INFO] Aplicacion finalizada correctamente.
)
