#!/bin/bash
"""
Pipeline Completo - Script de Ejecución

Este script ejecuta el pipeline completo de procesamiento:
1. Detección y extracción de personas (video_processor.py)
2. Mejora de calidad de imágenes (image_enhancer.py)
3. Aumentación de dataset de armas (simple_augmenter.py)
4. Generación de informe (generate_report.py)

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Octubre 2025
"""

echo "🚀 Pipeline Completo de Procesamiento de Imágenes"
echo "=" | tr '=' '=' | head -c 60; echo

# Verificar que existen los scripts necesarios
if [ ! -f "video_processor.py" ]; then
    echo "❌ Error: video_processor.py no encontrado"
    exit 1
fi

if [ ! -f "image_enhancer.py" ]; then
    echo "❌ Error: image_enhancer.py no encontrado"
    exit 1
fi

if [ ! -f "simple_augmenter.py" ]; then
    echo "❌ Error: simple_augmenter.py no encontrado"
    exit 1
fi

if [ ! -f "generate_report.py" ]; then
    echo "❌ Error: generate_report.py no encontrado"
    exit 1
fi

# Paso 1: Detección de personas (opcional, comentar si ya se ejecutó)
echo ""
echo "📹 Paso 1: Detección y Extracción de Personas"
echo "---"
echo "⚠️  Asegúrate de tener un video en input/"
echo "💡 Descomenta las siguientes líneas para ejecutar:"
echo "# python video_processor.py --video input/tu_video.mp4"
echo ""
read -p "¿Ya ejecutaste video_processor.py? (s/n): " respuesta1

if [ "$respuesta1" != "s" ]; then
    echo "⏭️  Saltando paso 1. Ejecuta manualmente: python video_processor.py --video input/tu_video.mp4"
fi

# Paso 2: Mejora de calidad de personas
echo ""
echo "🎨 Paso 2: Mejora de Calidad de Imágenes de Personas"
echo "---"
read -p "¿Ejecutar image_enhancer.py? (s/n): " respuesta2

if [ "$respuesta2" = "s" ]; then
    if [ -d "output/cropped_persons" ]; then
        echo "✅ Ejecutando image_enhancer.py..."
        python image_enhancer.py
    else
        echo "⚠️  Directorio output/cropped_persons no encontrado"
        echo "💡 Ejecuta primero video_processor.py"
    fi
else
    echo "⏭️  Saltando paso 2"
fi

# Paso 3: Aumentación de dataset de armas
echo ""
echo "🔄 Paso 3: Aumentación de Dataset de Armas"
echo "---"
read -p "¿Ejecutar simple_augmenter.py? (s/n): " respuesta3

if [ "$respuesta3" = "s" ]; then
    if [ -d "dataset/original" ]; then
        echo "✅ Ejecutando simple_augmenter.py..."
        python simple_augmenter.py
    else
        echo "⚠️  Directorio dataset/original no encontrado"
        echo "💡 Organiza tu dataset de armas en dataset/original/knife y dataset/original/pistol"
    fi
else
    echo "⏭️  Saltando paso 3"
fi

# Paso 4: Generación de informe
echo ""
echo "📊 Paso 4: Generación de Informe"
echo "---"
read -p "¿Ejecutar generate_report.py? (s/n): " respuesta4

if [ "$respuesta4" = "s" ]; then
    echo "✅ Ejecutando generate_report.py..."
    python generate_report.py
else
    echo "⏭️  Saltando paso 4"
fi

echo ""
echo "=" | tr '=' '=' | head -c 60; echo
echo "✅ Pipeline completo ejecutado"
echo ""
echo "📁 Resultados:"
echo "   - Personas extraídas: output/cropped_persons/"
echo "   - Personas mejoradas: output/enhanced_persons/"
echo "   - Dataset aumentado: dataset/augmented/"
echo "   - Informe: INFORME_DATASET.md"
echo ""
echo "💡 Próximo paso: Fine-tuning YOLOv8 para detección de armas (Stage 2)"
