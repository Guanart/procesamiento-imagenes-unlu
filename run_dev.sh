#!/bin/bash

# Script de desarrollo para ejecutar la aplicación Flask
# Uso: ./run_dev.sh

echo "🚀 Iniciando aplicación Flask para desarrollo..."

# Verificar si existe el directorio de uploads
if [ ! -d "uploads" ]; then
    echo "📁 Creando directorio uploads..."
    mkdir -p uploads
fi

# Verificar si existe .env
if [ ! -f ".env" ]; then
    echo "⚙️ Copiando configuración de ejemplo..."
    cp .env.example .env
    echo "   Edita el archivo .env con tus configuraciones"
fi

# Configurar variables de entorno para desarrollo
export FLASK_ENV=development
export FLASK_APP=app.py
export FLASK_DEBUG=1

echo "🔧 Configuración:"
echo "   FLASK_ENV: $FLASK_ENV"
echo "   FLASK_APP: $FLASK_APP"
echo "   Puerto: 5000"

echo ""
echo "📊 La aplicación estará disponible en:"
echo "   http://localhost:5000"
echo ""
echo "🔍 API endpoints:"
echo "   POST /upload         - Subir imagen via web"
echo "   POST /api/analyze    - Análisis via API"
echo ""
echo "⏹️ Presiona Ctrl+C para detener"
echo ""

# TODO: Verificar si las dependencias están instaladas
# python -c "import flask, PIL, numpy" 2>/dev/null || {
#     echo "❌ Dependencias faltantes. Ejecuta: pip install -r requirements.txt"
#     exit 1
# }

# Ejecutar la aplicación
python app.py