# Person Extraction

Pipeline de extracción y mejora de personas desde videos usando YOLOv8.

## Descripción

Pipeline de 2 pasos:
1. **Extracción:** Detecta y extrae personas usando YOLOv8n (modelo COCO)
2. **Mejora:** Aplica 5 técnicas de mejora de calidad de imagen

## Uso

### 1. Extraer Personas
```bash
python video_processor.py
```
- Input: `input/video.mp4`
- Output: `output/cropped_persons/` (270 imágenes @ 1 fps)

### 2. Mejorar Calidad
```bash
# Versión avanzada (5 técnicas)
python image_enhancer.py

# Versión básica (teoría confirmada)
python image_enhancer_basic.py
```
- Input: `output/cropped_persons/`
- Output: `output/enhanced_persons/` (640×640 normalizadas)

### 3. Generar Reporte (Opcional)
```bash
python generate_report.py
```

## Técnicas de Mejora

**Avanzada:**
- Interpolación spline (bicubic)
- Bilateral filter (reducción ruido)
- Unsharp masking (nitidez)
- CLAHE (contraste adaptativo)
- Canny (detección bordes)

**Básica:**
- Bicubic interpolation
- Gaussian blur
- Laplacian sharpening
- Histogram equalization
- Sobel edge detection

## Estructura

```
person_extraction/
├── video_processor.py          # Paso 1: Extracción
├── image_enhancer.py           # Paso 2: Mejora avanzada
├── image_enhancer_basic.py     # Paso 2: Mejora básica
├── generate_report.py          # Análisis dataset
├── run_pipeline.sh             # Automatización
├── input/                      # Videos de entrada
└── output/
    ├── cropped_persons/        # Personas extraídas
    └── enhanced_persons/       # Personas mejoradas
```

## Resultados

- **Video:** 13.64s @ 25 FPS → 341 frames totales
- **Muestreo:** 1 frame/segundo → 13 frames procesados
- **Personas:** 270 extraídas (confianza > 0.50)
- **Normalización:** 640×640 píxeles
