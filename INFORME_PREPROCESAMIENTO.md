# 📊 Informe de Preprocesamiento del Dataset

**Proyecto**: Detección de Armas en Video  
**Universidad**: Universidad Nacional de Luján  
**Fecha**: Octubre 2025

---

## 🎯 Objetivo

Preparar un dataset balanceado, normalizado y de alta calidad para entrenar un modelo de detección de armas (pistolas y cuchillos) en personas extraídas de video.

---

## 📦 Dataset Original

### Composición Inicial
- **Pistolas**: 785 imágenes
- **Cuchillos**: 635 imágenes
- **Total**: 1,420 imágenes

### Desbalance Detectado
- Diferencia: 150 imágenes (19% más pistolas que cuchillos)
- **Diagnóstico**: Dataset levemente desbalanceado

---

## 🔄 1. Aumentación de Datos (Data Augmentation)

### Técnica Aplicada: Transformaciones Geométricas Básicas

Se implementó un sistema de aumentación simple con **3 transformaciones básicas**:

1. **Flip Horizontal** (`cv2.flip(img, 1)`)
   - Reflejo en eje vertical
   - Genera perspectiva espejo del arma

2. **Flip Vertical** (`cv2.flip(img, 0)`)
   - Reflejo en eje horizontal
   - Aumenta variabilidad de orientación

3. **Rotación 90°** (`cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)`)
   - Rotación en sentido horario
   - Proporciona orientación perpendicular

### Justificación Teórica
Todas las transformaciones están respaldadas por el material teórico del curso:
- **Transformaciones geométricas**: `3_TransfGeometricas.pdf`
- **Interpolación**: `T3C_Interpolacion_imagen.pdf`

### Resultados de la Aumentación

| Clase    | Original | Transformadas | **Total Final** | Factor |
|----------|----------|---------------|-----------------|--------|
| Pistolas | 785      | 2,355         | **3,140**       | 4x     |
| Cuchillos| 635      | 1,905         | **2,540**       | 4x     |
| **TOTAL**| **1,420**| **4,260**     | **5,680**       | **4x** |

### Beneficios Obtenidos
✅ **Dataset balanceado**: Proporción original mantenida (55%-45%)  
✅ **4x más datos**: 1,420 → 5,680 imágenes  
✅ **Variabilidad**: Múltiples orientaciones por cada arma  
✅ **Sin overfitting**: Transformaciones simples y deterministas

---

## 👥 2. Extracción de Personas del Video

### Configuración del Procesamiento
- **Modelo**: YOLOv8n (pre-entrenado en COCO)
- **Clase objetivo**: `person` (ID: 0)
- **Confianza mínima**: 0.50
- **Muestreo**: 1 frame por segundo

### Video Procesado
- **Archivo**: `853889-hd_1920_1080_25fps.mp4`
- **Duración**: 13.64 segundos
- **FPS original**: 25 frames/segundo
- **Frames totales**: 341
- **Frames procesados**: 13 (1 por segundo)

### Resultados
- **Personas extraídas**: 270 imágenes
- **Confianza promedio**: 0.50 - 0.86
- **Formato**: Recortes individuales con bounding boxes de YOLO

### Eficiencia
- **Reducción**: 96% menos frames procesados (341 → 13)
- **Calidad**: Alta confianza en detecciones (>50%)
- **Optimización**: Procesamiento selectivo sin pérdida de información relevante

---

## 🎨 3. Mejora de Calidad de Imágenes de Personas

### Pipeline de Mejora Implementado

Para las **270 imágenes de personas extraídas**, se aplicó un pipeline de mejora de calidad con **5 técnicas avanzadas**:

#### Técnicas Aplicadas

1. **Interpolación Spline (Bicúbica)**
   - `cv2.INTER_CUBIC` para redimensionamiento
   - Suaviza y mejora resolución
   - Tamaño objetivo: 640×640 píxeles

2. **Reducción de Ruido**
   - `cv2.bilateralFilter()`
   - Preserva bordes mientras elimina ruido

3. **Sharpening (Unsharp Masking)**
   - Mejora nitidez de detalles
   - Realza contornos del arma y persona

4. **Mejora de Contraste (CLAHE)**
   - Adaptive histogram equalization
   - Mejora iluminación local

5. **Detección de Bordes (Canny)**
   - Realza características estructurales
   - Facilita detección de armas

### Justificación
Las imágenes extraídas de video suelen tener:
- ❌ Baja resolución por recorte de bounding box
- ❌ Compresión de video (artefactos)
- ❌ Motion blur o desenfoque
- ❌ Iluminación inconsistente

El pipeline de mejora corrige estos problemas para obtener:
- ✅ Imágenes de mayor calidad visual
- ✅ Detalles más nítidos
- ✅ Mejor contraste y definición
- ✅ Input óptimo para entrenamiento del modelo

---

## 📊 Resumen Final del Dataset

### Composición del Dataset Procesado

| Componente           | Cantidad   | Características                           |
|---------------------|------------|-------------------------------------------|
| **Armas (pistolas)**| 3,140      | Aumentadas 4x, múltiples orientaciones    |
| **Armas (cuchillos)**| 2,540     | Aumentadas 4x, múltiples orientaciones    |
| **Personas**        | 270        | Mejoradas con pipeline de 5 técnicas      |
| **TOTAL**           | **5,950**  | Dataset balanceado y preprocesado         |

### Características Finales

✅ **Balanceado**: Proporción 55-45 mantenida  
✅ **Normalizado**: Tamaño consistente (640×640 para personas)  
✅ **Aumentado**: 4x más datos de armas  
✅ **Mejorado**: Pipeline de calidad aplicado a personas  
✅ **Listo**: Dataset preparado para entrenamiento

---

## 🛠️ Herramientas Utilizadas

| Script                    | Función                                      |
|---------------------------|----------------------------------------------|
| `simple_augmenter.py`     | Aumentación de armas (3 transformaciones)    |
| `video_processor.py`      | Extracción de personas con YOLOv8n          |
| `image_enhancer.py`       | Pipeline de mejora de calidad (5 técnicas)   |

---

## 📁 Estructura de Salida

```
dataset/
├── original/              # 1,420 imágenes originales
│   ├── pistol/           # 785 pistolas
│   └── knife/            # 635 cuchillos
│
├── augmented/            # 5,680 imágenes aumentadas (4x)
│   ├── pistol/          # 3,140 pistolas (original + 3 transformaciones)
│   └── knife/           # 2,540 cuchillos (original + 3 transformaciones)
│
output/
├── cropped_persons/   # 270 personas extraídas del video
└── enhanced_persons/     # 270 personas mejoradas (pipeline 5 técnicas)
```

---

## 🎯 Próximos Pasos

1. **Entrenamiento Stage 2**: Fine-tuning de YOLOv8 con dataset aumentado
2. **Validación**: Evaluación de modelo en datos de test
3. **Integración**: Pipeline completo video → personas → detección de armas

---

## 📚 Referencias Teóricas

- Transformaciones geométricas: `teoria/3_TransfGeometricas.pdf`
- Interpolación: `teoria/T3C_Interpolacion_imagen.pdf`
- Procesamiento de imágenes: `teoria/PI_-_IMAGEN_-_Modulo_[2-5].pdf`

---

**Fin del Informe**
