@echo off
title Instalar Dependencias - Tarea 3
echo ==========================================
echo    INSTALADOR DE DEPENDENCIAS TAREA 3
echo ==========================================
echo.

REM 1. Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado en el PATH.
    echo Por favor instale Python 3.9 o superior y asegurese de agregarlo al PATH.
    pause
    exit /b 1
)

echo [INFO] Python encontrado.
python --version

REM 2. Verificar/Crear Entorno Virtual
if exist ".venv" (
    echo [INFO] Entorno virtual '.venv' detectado.
) else (
    echo [INFO] Creando entorno virtual '.venv'...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Fallo al crear el entorno virtual.
        pause
        exit /b 1
    )
)

REM 3. Instalar Dependencias
echo [INFO] Instalando dependencias desde requirements.txt...
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Hubo un problema instalando las dependencias.
        pause
        exit /b 1
    )
) else (
    echo [ERROR] No se encontro el archivo requirements.txt.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo    INSTALACION COMPLETADA EXITOSAMENTE
echo ==========================================
echo.
echo Para iniciar la aplicacion, ejecute 'run_app.bat'
echo.
pause
