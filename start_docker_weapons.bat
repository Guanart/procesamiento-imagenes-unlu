@echo off
REM Script para iniciar el Sistema de Detección de Armas con Docker (Windows)

echo ==========================================
echo   Sistema de Deteccion de Armas - Docker
echo ==========================================
echo.

REM Verificar Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker no esta instalado
    echo Instalalo desde: https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

echo Docker detectado
echo.

REM Verificar modelo de armas
if not exist "..\weapons_detector2\results_light\best_model.pth" (
    echo Error: Modelo de armas no encontrado
    echo Ubicacion esperada: weapons_detector2\results_light\best_model.pth
    echo.
    echo Entrena el modelo primero:
    echo python weapons_detector2\train_fasterrcnn_light.py --amp
    pause
    exit /b 1
)

echo Modelo de armas encontrado
echo.

REM Verificar modelo YOLO
if not exist "..\person_extraction\yolov8n.pt" (
    echo Descargando modelo YOLOv8...
    if not exist "..\person_extraction" mkdir "..\person_extraction"
    cd ..\person_extraction
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    cd ..\flask_analyzer
)

echo Modelo YOLO encontrado
echo.

REM Crear directorios
if not exist "uploads\weapons" mkdir uploads\weapons
if not exist "results\weapons" mkdir results\weapons

echo Construyendo imagen Docker...
docker-compose build

if errorlevel 1 (
    echo Error al construir la imagen
    pause
    exit /b 1
)

echo.
echo Imagen construida exitosamente
echo.

echo Selecciona una opcion:
echo 1^) Iniciar contenedor ^(modo detached^)
echo 2^) Iniciar contenedor ^(modo interactivo - ver logs^)
echo 3^) Detener contenedor
echo 4^) Ver logs
echo 5^) Reiniciar contenedor
echo 6^) Eliminar contenedor e imagen
echo 7^) Salir
echo.
set /p option="Opcion [1-7]: "

if "%option%"=="1" goto start_detached
if "%option%"=="2" goto start_interactive
if "%option%"=="3" goto stop
if "%option%"=="4" goto logs
if "%option%"=="5" goto restart
if "%option%"=="6" goto clean
if "%option%"=="7" goto exit
goto invalid

:start_detached
echo.
echo Iniciando contenedor en segundo plano...
docker-compose up -d
if not errorlevel 1 (
    echo.
    echo Contenedor iniciado
    echo.
    echo Accede a: http://localhost:5001
    echo.
    echo Para ver logs: docker-compose logs -f
    echo Para detener: docker-compose down
)
goto end

:start_interactive
echo.
echo Iniciando contenedor ^(presiona Ctrl+C para detener^)...
docker-compose up
goto end

:stop
echo.
echo Deteniendo contenedor...
docker-compose down
echo Contenedor detenido
goto end

:logs
echo.
echo Logs del contenedor ^(presiona Ctrl+C para salir^):
echo.
docker-compose logs -f
goto end

:restart
echo.
echo Reiniciando contenedor...
docker-compose restart
echo Contenedor reiniciado
goto end

:clean
echo.
set /p confirm="Estas seguro? Esto eliminara el contenedor y la imagen [s/N]: "
if /i "%confirm%"=="s" (
    docker-compose down
    docker rmi flask_analyzer-weapon-detector 2>nul
    echo Limpieza completada
) else (
    echo Operacion cancelada
)
goto end

:exit
echo Saliendo...
exit /b 0

:invalid
echo Opcion invalida
pause
exit /b 1

:end
echo.
echo Presiona cualquier tecla para salir...
pause >nul
