# 📚 Documentación Técnica - Pipeline de Mejora de Imágenes

## 🎯 Objetivo

Mejorar la calidad de las imágenes de personas extraídas por `video_processor.py` para facilitar la detección de armas en el Stage 2 del pipeline.

## 🔧 Componentes Implementados

### 1. `image_enhancer.py` - Pipeline de Mejora de Calidad

**Clase Principal**: `ImageEnhancer`

**Técnicas Aplicadas**:

#### 1.1 Interpolación Spline Cúbica (`resize_with_spline`)
- **Método**: `cv2.INTER_CUBIC`
- **Objetivo**: Redimensionar imágenes pequeñas manteniendo calidad
- **Tamaño mínimo**: 200 filas × 100 columnas
- **Preserva**: Aspect ratio original
- **Ventaja**: La interpolación cúbica spline produce resultados más suaves que la interpolación lineal o nearest-neighbor

#### 1.2 Reducción de Ruido (`reduce_noise`)
- **Método**: Filtro Bilateral (`cv2.bilateralFilter`)
- **Parámetros**: 
  - `d=9`: Diámetro del vecindario
  - `sigmaColor=75`: Suavizado en espacio de color
  - `sigmaSpace=75`: Suavizado en espacio espacial
- **Ventaja**: Preserva bordes mientras reduce ruido de fondo

#### 1.3 Realce de Nitidez (`sharpen_image`)
- **Método**: Unsharp Masking
- **Fórmula**: `imagen_nitida = 1.5 × imagen - 0.5 × blur`
- **Proceso**:
  1. Crear versión borrosa con Gaussian blur
  2. Restar blur de la original con pesos ajustados
- **Ventaja**: Aumenta los detalles finos y mejora la percepción visual

#### 1.4 Mejora de Contraste (`enhance_contrast`)
- **Método**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Parámetros**:
  - `clipLimit=2.0`: Límite de contraste
  - `tileGridSize=(8,8)`: Tamaño de cuadrícula
- **Proceso**:
  1. Convertir a espacio LAB
  2. Aplicar CLAHE solo al canal L (luminosidad)
  3. Reconstruir imagen BGR
- **Ventaja**: Mejora contraste localmente sin saturar

#### 1.5 Realce de Bordes (`enhance_edges`)
- **Método**: Detección Canny + Combinación Ponderada
- **Proceso**:
  1. Detectar bordes con Canny (umbrales 50-150)
  2. Dilatar bordes para hacerlos más visibles
  3. Combinar con imagen original (90% imagen + 10% bordes)
- **Ventaja**: Mejora la definición de contornos sin perder información de color

### 2. `generate_report.py` - Generador de Informes

**Clase Principal**: `DatasetAnalyzer`

**Análisis Realizados**:

#### 2.1 Dataset de Armas
- **Balanceo de clases**: Verifica equilibrio entre knife/pistol
- **Normalización de tamaños**: Analiza distribución de dimensiones
- **Detección de augmentation**: Identifica transformaciones aplicadas
- **Estadísticas**: Min, max, promedio de dimensiones

#### 2.2 Dataset de Personas
- **Conteo y dimensiones**: Análisis de imágenes originales
- **Verificación de mejoras**: Detecta si se aplicó el pipeline
- **Estado del Stage 2**: Verifica preparación para detección de armas

#### 2.3 Formatos de Salida
- **Markdown** (`.md`): Informe legible para humanos
- **JSON** (`.json`): Datos estructurados para procesamiento automatizado

## 📊 Flujo de Trabajo Recomendado

```
┌─────────────────────┐
│  1. video_processor │  ← Extracción de personas de videos
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  output/cropped_    │  ← Personas extraídas (baja calidad)
│  persons/           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. image_enhancer  │  ← Pipeline de mejora de calidad
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  output/enhanced_   │  ← Personas mejoradas (alta calidad)
│  persons/           │
└──────────┬──────────┘
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│  3a. simple_        │    │  3b. generate_      │
│      augmenter      │    │      report         │
│  (para armas)       │    │  (análisis)         │
└──────────┬──────────┘    └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  dataset/augmented/ │  ← Dataset de armas aumentado
└─────────────────────┘
```

## 🎓 Conceptos Técnicos

### Interpolación Spline Cúbica
- **Definición**: Método de interpolación que usa polinomios cúbicos por partes
- **Ventaja**: Produce curvas suaves y continuas
- **Aplicación**: Redimensionamiento de imágenes con calidad superior a la interpolación lineal
- **En OpenCV**: `cv2.INTER_CUBIC`

### Filtro Bilateral
- **Definición**: Filtro no lineal que preserva bordes
- **Funcionamiento**: Combina proximidad espacial y similitud de intensidad
- **Ventaja**: Reduce ruido sin difuminar bordes importantes
- **Aplicación**: Preprocessing antes de detección de objetos

### Unsharp Masking
- **Definición**: Técnica de realce de nitidez
- **Funcionamiento**: Resta una versión borrosa de la original
- **Efecto**: Aumenta los detalles finos y la percepción de nitidez
- **Historia**: Técnica tradicional de fotografía analógica

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Definición**: Ecualización de histograma adaptativa con límite de contraste
- **Ventaja sobre HE simple**: Evita sobre-amplificación de ruido
- **Funcionamiento**: Divide la imagen en tiles y ecualiza cada uno
- **Aplicación**: Mejora de imágenes médicas, visión nocturna, etc.

### Detección de Bordes Canny
- **Definición**: Algoritmo de detección de bordes multi-etapa
- **Pasos**:
  1. Suavizado Gaussiano (reducción de ruido)
  2. Cálculo de gradientes
  3. Supresión no-máxima
  4. Umbralización por histéresis
- **Ventaja**: Detecta bordes precisos con buena localización

## 📈 Parámetros Configurables

### `image_enhancer.py`
```bash
--input DIR           # Directorio de entrada (default: output/cropped_persons)
--output DIR          # Directorio de salida (default: output/enhanced_persons)
--min-height PIXELS   # Altura mínima objetivo (default: 200)
--min-width PIXELS    # Ancho mínimo objetivo (default: 100)
```

### `generate_report.py`
```bash
--weapons-dir DIR     # Directorio de armas (default: dataset/augmented)
--persons-dir DIR     # Directorio de personas originales
--enhanced-dir DIR    # Directorio de personas mejoradas
--output FILE         # Archivo de informe (default: INFORME_DATASET.md)
```

## 🔬 Validación de Resultados

### Métricas de Calidad
- **Tamaño mínimo alcanzado**: Todas las imágenes ≥ 200×100 px
- **Reducción de ruido**: Visual, sin medida cuantitativa automática
- **Nitidez**: Aumento perceptual de detalles finos
- **Contraste**: Mejor distribución de intensidades (verificable con histograma)
- **Definición de bordes**: Contornos más marcados

### Verificación Manual
1. Comparar imágenes antes/después visualmente
2. Verificar que no hay sobre-procesamiento (artefactos, halo effects)
3. Comprobar que los detalles importantes se preservan
4. Evaluar si las mejoras facilitan la detección en Stage 2

## 🚀 Próximos Pasos

### Stage 2: Detección de Armas en Personas
1. **Fine-tuning YOLOv8**: Entrenar modelo personalizado
2. **Dataset**: Combinar personas mejoradas + dataset de armas aumentado
3. **Anotación**: Etiquetar armas en imágenes de personas
4. **Entrenamiento**: Transfer learning desde YOLOv8n pre-entrenado
5. **Evaluación**: mAP, precisión, recall en conjunto de validación

### Mejoras Futuras del Pipeline
- **Super-Resolution**: Usar modelos de deep learning (ESRGAN, Real-ESRGAN)
- **Denoising avanzado**: Modelos basados en CNN (DnCNN, FFDNet)
- **Detección de blur**: Filtrar imágenes muy borrosas antes del procesamiento
- **Normalización automática**: Ajuste de iluminación y color

## 📖 Referencias

- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
- **CLAHE**: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- **Bilateral Filter**: Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for gray and color images"
- **Canny Edge Detection**: Canny, J. (1986). "A Computational Approach to Edge Detection"

---

*Documento técnico generado para el proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján, Octubre 2025*
