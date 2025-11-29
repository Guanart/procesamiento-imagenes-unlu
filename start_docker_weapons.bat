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

REM Detectar comando docker compose
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose version >nul 2>&1
    if errorlevel 1 (
        echo Error: Docker Compose no esta instalado
        pause
        exit /b 1
    ) else (
        set "DOCKER_COMPOSE=docker-compose"
    )
) else (
    set "DOCKER_COMPOSE=docker compose"
)

REM Verificar modelo de armas
set "MODEL_PATH=models\weapon_detection\best_model.pth"
if not exist "%MODEL_PATH%" (
    echo Error: Modelo de armas no encontrado
    echo Ubicacion esperada: %MODEL_PATH%
    echo.
    echo Entrena el modelo primero:
    echo cd src\weapon_detection\training ^&^& python pipeline.py --skip-stages split augment
    pause
    exit /b 1
)

echo Modelo de armas encontrado
echo.

REM Verificar modelo YOLO
set "YOLO_PATH=src\person_extraction\yolov8n.pt"
if not exist "%YOLO_PATH%" (
    echo Descargando modelo YOLOv8...
    if not exist "src\person_extraction" mkdir "src\person_extraction"
    pushd src\person_extraction
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    popd
)

echo Modelo YOLO encontrado
echo.

REM Crear directorios
if not exist "apps\image_lab\uploads" mkdir apps\image_lab\uploads
if not exist "apps\weapon_monitor\uploads\weapons" mkdir apps\weapon_monitor\uploads\weapons
if not exist "apps\weapon_monitor\results\weapons" mkdir apps\weapon_monitor\results\weapons

echo Construyendo imagen Docker...
call :compose build

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
call :compose up -d
if not errorlevel 1 (
    echo.
    echo Contenedor iniciado
    echo.
    echo Accede a:
    echo   - http://localhost:5000 ^(Image Lab^)
    echo   - http://localhost:5001 ^(Weapon Monitor^)
    echo.
    echo Para ver logs: %DOCKER_COMPOSE% logs -f
    echo Para detener: %DOCKER_COMPOSE% down
)
goto end

:start_interactive
echo.
echo Iniciando contenedor ^(presiona Ctrl+C para detener^)...
call :compose up
goto end

:stop
echo.
echo Deteniendo contenedor...
call :compose down
echo Contenedor detenido
goto end

:logs
echo.
echo Logs del contenedor ^(presiona Ctrl+C para salir^):
echo.
call :compose logs -f
goto end

:restart
echo.
echo Reiniciando contenedor...
call :compose restart
echo Contenedor reiniciado
goto end

:clean
echo.
set /p confirm="Estas seguro? Esto eliminara el contenedor y la imagen [s/N]: "
if /i "%confirm%"=="s" (
    call :compose down
    docker rmi weapon-monitor image-lab 2>nul
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

goto :eof

:compose
if "%DOCKER_COMPOSE%"=="docker compose" (
    docker compose %*
) else (
    docker-compose %*
)
exit /b %errorlevel%
