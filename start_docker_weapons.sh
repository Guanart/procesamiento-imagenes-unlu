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

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: Docker Compose no está instalado"
    exit 1
fi

echo "✅ Docker detectado"
echo ""

# Verificar que existe el modelo entrenado
if [ ! -f "weapons_detector2/results_light/best_model.pth" ]; then
    echo "❌ Error: Modelo de armas no encontrado"
    echo "   Ubicación esperada: weapons_detector2/results_light/best_model.pth"
    echo ""
    echo "   Por favor, entrena el modelo primero:"
    echo "   python weapons_detector2/train_fasterrcnn_light.py --amp"
    exit 1
fi

echo "✅ Modelo de armas encontrado"
echo ""

# Verificar/descargar modelo YOLO
if [ ! -f "person_extraction/yolov8n.pt" ]; then
    echo "📥 Descargando modelo YOLOv8..."
    mkdir -p person_extraction
    cd person_extraction
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || \
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    cd ..
fi

echo "✅ Modelo YOLO encontrado"
echo ""

# Crear directorios necesarios
mkdir -p flask_analyzer/uploads/weapons
mkdir -p flask_analyzer/results/weapons

echo "🔨 Construyendo imagen Docker..."
docker compose build

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
        docker-compose up -d
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Contenedor iniciado"
            echo ""
            echo "📱 Accede a: http://localhost:5001"
            echo ""
            echo "Para ver logs: docker-compose logs -f"
            echo "Para detener: docker-compose down"
        fi
        ;;
    2)
        echo ""
        echo "🚀 Iniciando contenedor (presiona Ctrl+C para detener)..."
        docker-compose up
        ;;
    3)
        echo ""
        echo "⏹ Deteniendo contenedor..."
        docker-compose down
        echo "✅ Contenedor detenido"
        ;;
    4)
        echo ""
        echo "📋 Logs del contenedor (presiona Ctrl+C para salir):"
        echo ""
        docker-compose logs -f
        ;;
    5)
        echo ""
        echo "🔄 Reiniciando contenedor..."
        docker-compose restart
        echo "✅ Contenedor reiniciado"
        ;;
    6)
        echo ""
        read -p "⚠️  ¿Estás seguro? Esto eliminará el contenedor y la imagen [s/N]: " confirm
        if [ "$confirm" = "s" ] || [ "$confirm" = "S" ]; then
            docker-compose down
            docker rmi flask_analyzer-weapon-detector 2>/dev/null
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
