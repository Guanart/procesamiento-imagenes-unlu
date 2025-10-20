Paso 1: Verificar estructura de directorios y sus imagenes

echo "🔍 Contando imágenes en dataset/original/..." && find dataset/original -type f \( -name "*.jpg" -o -name "*.png" \) | head -10 && echo "" && echo "Total de imágenes en dataset/original:" && find dataset/original -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l

echo "📊 Distribución por clase:" && echo "Pistolas:" && find dataset/original/pistol -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l && echo "Cuchillos:" && find dataset/original/knife -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo "0"

echo "👥 Contando personas extraídas..." && ls output/cropped_persons/*.jpg 2>/dev/null | wc -l

Paso 2: Aplicar Data Augmentation al dataset de armas

echo "🔄 Ejecutando Data Augmentation..." && python simple_augmenter.py --input dataset/original --output dataset/augmented
