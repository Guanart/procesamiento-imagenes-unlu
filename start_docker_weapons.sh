#!/bin/bash
# Script para iniciar el Sistema de Detección de Armas con Docker

echo "🔫 Sistema de Detección de Armas - Docker"
echo "=========================================="
echo ""

# Verificar que Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker no está instalado"
    echo "   Instálalo desde: https://docs.docker.com/get-docker/"
    exit 1
fi

if docker compose version &> /dev/null; then
    DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE=(docker-compose)
else
    echo "❌ Error: Docker Compose no está instalado"
    exit 1
fi

echo "✅ Docker detectado"
echo ""

# Verificar que existe el modelo entrenado
MODEL_PATH="models/weapon_detection/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Modelo de armas no encontrado"
    echo "   Ubicación esperada: $MODEL_PATH"
    echo ""
    echo "   Por favor, entrena el modelo primero:"
    echo "   cd src/weapon_detection/training && python pipeline.py --skip-stages split augment"
    exit 1
fi

echo "✅ Modelo de armas encontrado"
echo ""

# Verificar/descargar modelo YOLO
YOLO_PATH="src/person_extraction/yolov8n.pt"
if [ ! -f "$YOLO_PATH" ]; then
    echo "📥 Descargando modelo YOLOv8..."
    mkdir -p src/person_extraction
    pushd src/person_extraction > /dev/null
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || \
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    popd > /dev/null
fi

echo "✅ Modelo YOLO encontrado"
echo ""

# Crear directorios necesarios
mkdir -p apps/image_lab/uploads
mkdir -p apps/weapon_monitor/uploads/weapons
mkdir -p apps/weapon_monitor/results/weapons

echo "🔨 Construyendo imagen Docker..."
"${DOCKER_COMPOSE[@]}" build

if [ $? -ne 0 ]; then
    echo "❌ Error al construir la imagen"
    exit 1
fi

echo ""
echo "✅ Imagen construida exitosamente"
echo ""

# Menú de opciones
echo "Selecciona una opción:"
echo "1) Iniciar contenedor (modo detached)"
echo "2) Iniciar contenedor (modo interactivo - ver logs)"
echo "3) Detener contenedor"
echo "4) Ver logs"
echo "5) Reiniciar contenedor"
echo "6) Eliminar contenedor e imagen"
echo "7) Salir"
echo ""
read -p "Opción [1-7]: " option

case $option in
    1)
        echo ""
        echo "🚀 Iniciando contenedor en segundo plano..."
        "${DOCKER_COMPOSE[@]}" up -d
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Contenedor iniciado"
            echo ""
            echo "📱 Accede a:"
            echo "   - http://localhost:5000 (Image Lab)"
            echo "   - http://localhost:5001 (Weapon Monitor)"
            echo ""
            echo "Para ver logs: ${DOCKER_COMPOSE[*]} logs -f"
            echo "Para detener: ${DOCKER_COMPOSE[*]} down"
        fi
        ;;
    2)
        echo ""
        echo "🚀 Iniciando contenedor (presiona Ctrl+C para detener)..."
        "${DOCKER_COMPOSE[@]}" up
        ;;
    3)
        echo ""
        echo "⏹ Deteniendo contenedor..."
        "${DOCKER_COMPOSE[@]}" down
        echo "✅ Contenedor detenido"
        ;;
    4)
        echo ""
        echo "📋 Logs del contenedor (presiona Ctrl+C para salir):"
        echo ""
        "${DOCKER_COMPOSE[@]}" logs -f
        ;;
    5)
        echo ""
        echo "🔄 Reiniciando contenedor..."
        "${DOCKER_COMPOSE[@]}" restart
        echo "✅ Contenedor reiniciado"
        ;;
    6)
        echo ""
        read -p "⚠️  ¿Estás seguro? Esto eliminará el contenedor y la imagen [s/N]: " confirm
        if [ "$confirm" = "s" ] || [ "$confirm" = "S" ]; then
            "${DOCKER_COMPOSE[@]}" down
            docker rmi weapon-monitor image-lab 2>/dev/null
            echo "✅ Limpieza completada"
        else
            echo "Operación cancelada"
        fi
        ;;
    7)
        echo "👋 Saliendo..."
        exit 0
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac
