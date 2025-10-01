# Stage 1: Detección de Personas y Extracción de Imágenes

## 📋 Descripción del Proyecto

Este proyecto implementa el **Stage 1** de un sistema de detección de riesgo en dos etapas. El objetivo principal es procesar videos y extraer imágenes de todas las personas detectadas, preparando el dataset para una segunda fase de análisis enfocada en la detección de armas.

### 🎯 Objetivo

Desarrollar un sistema automatizado que:
- Procese videos frame por frame
- Detecte personas usando inteligencia artificial
- Extraiga y guarde imágenes recortadas de cada persona
- Prepare el dataset para el Stage 2 (detección de armas)

---

## 🧠 Fundamentación del Modelo y Arquitectura

### Modelo Seleccionado: YOLOv8n

**YOLOv8n** (You Only Look Once - versión nano) ha sido seleccionado como el modelo base para este proyecto por las siguientes razones técnicas:

#### Características Técnicas:
- **Arquitectura**: Red Neuronal Convolucional (CNN) de una sola etapa (*single-stage*)
- **Eficiencia**: Versión nano optimizada para velocidad y bajo consumo de recursos
- **Precisión**: Mantiene alta accuracy mientras reduce el tiempo de inferencia
- **Fuente**: [Repositorio oficial de Ultralytics en GitHub](https://github.com/ultralytics/ultralytics)

#### Ventajas de YOLOv8n:
1. **Detección en Tiempo Real**: Capaz de procesar múltiples frames por segundo
2. **Arquitectura Unificada**: Un solo modelo maneja localización y clasificación simultáneamente
3. **Optimización de Recursos**: La versión nano requiere menos memoria y procesamiento
4. **Pre-entrenamiento COCO**: Ya optimizado para detectar la clase 'persona'

### Tipo de Red Neural

**Arquitectura CNN de Una Sola Etapa (Single-Stage)**:
- **Ventaja Principal**: Realiza detección y localización en una sola pasada
- **Eficiencia**: Más rápido que arquitecturas de dos etapas (R-CNN, Fast R-CNN)
- **Aplicabilidad**: Ideal para procesamiento de video en tiempo real

---

## 📊 Justificación del Pre-entrenamiento: COCO vs. ImageNet

### ¿Por qué COCO Dataset?

La elección del dataset COCO sobre ImageNet se fundamenta en los siguientes aspectos técnicos:

#### **COCO Dataset - La Elección Correcta**:
- **Tarea Específica**: Entrenado para **Detección de Objetos** (Object Detection)
- **Localización Espacial**: Incluye coordenadas de *bounding boxes*
- **Clase 'Person'**: La clase `person` (ID: 0) está altamente optimizada
- **Diversidad**: 80 clases de objetos cotidianos con variaciones contextuales
- **Robustez**: Entrenado con imágenes en múltiples condiciones (iluminación, ángulos, oclusiones)

#### **ImageNet - Limitaciones para Nuestro Caso**:
- **Tarea Diferente**: Enfocado en **Clasificación de Imágenes** solamente
- **Sin Localización**: No proporciona información espacial de objetos
- **Inadecuado**: Requeriría arquitectura adicional para detección

### Beneficios Específicos para el Stage 1:

1. **Optimización Inmediata**: El modelo YOLOv8n + COCO ya está optimizado para detectar personas
2. **Sin Reentrenamiento**: No requiere fine-tuning para el Stage 1
3. **Alta Precisión**: Accuracy superior al 90% en detección de personas
4. **Preparación Stage 2**: Las imágenes extraídas son ideales para entrenar detectores de armas

---

## 🔄 Metodología y Preparación para el Stage 2

### Flujo de Trabajo del Stage 1:

```
Video de Entrada → Extracción de Frames → YOLOv8 Inferencia → Filtrado ('person' solamente) → Recorte de Bounding Boxes → Almacenamiento Organizado
```

#### Proceso Detallado:

1. **Lectura de Video**: 
   - Procesamiento frame por frame usando OpenCV
   - Mantenimiento de metadatos (número de frame, timestamp)

2. **Inferencia YOLOv8**:
   - Detección de todos los objetos en cada frame
   - Filtrado específico para clase `person` (ID: 0)
   - Aplicación de umbral de confianza (>0.5)

3. **Extracción y Recorte**:
   - Cálculo de coordenadas de bounding box
   - Recorte preciso de la región de la persona
   - Validación de dimensiones mínimas

4. **Almacenamiento Estructurado**:
   - Nomenclatura sistemática: `frame_XXXXXX_person_X_conf_X.XX.jpg`
   - Organización por confianza y frame de origen
   - Metadatos preservados en el nombre del archivo

### Dataset de Salida para Stage 2:

Las imágenes generadas por el Stage 1 constituyen el **Dataset de Entrada** para el **Stage 2**, donde se realizará:

- **Fine-tuning** del modelo para detectar objetos específicos (`pistola`, `cuchillo`)
- **Transfer Learning** desde detección de personas a detección de armas
- **Entrenamiento Supervisado** con las imágenes de personas como input

#### Ventajas del Dataset Generado:
- **Relevancia Contextual**: Solo contiene personas (contexto apropiado para armas)
- **Diversidad Visual**: Diferentes poses, ángulos y condiciones
- **Calidad Controlada**: Filtrado por confianza garantiza calidad mínima
- **Escalabilidad**: Procesamiento automatizado permite grandes volúmenes

---

## 🚀 Instrucciones de Uso

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **GPU**: Recomendada para procesamiento rápido (CUDA compatible)
- **RAM**: Mínimo 4GB, recomendado 8GB
- **Espacio**: Dependiente del tamaño del video y personas detectadas

### Instalación

1. **Clonar/Descargar el proyecto**:
   ```bash
   cd procesamiento-imagenes-unlu
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Las dependencias principales incluyen:
   - `ultralytics>=8.0.0` - Framework YOLOv8
   - `opencv-python>=4.8.0` - Procesamiento de video
   - `torch>=2.0.0` - Backend de deep learning
   - `torchvision>=0.15.0` - Utilidades de visión por computadora

3. **Preparar video de entrada**:
   ```bash
   mkdir -p input
   # Colocar video en input/test_video.mp4 o especificar ruta
   ```

### Ejecución

#### Uso Básico:
```bash
python video_processor.py
```

#### Uso Avanzado:
```bash
# Especificar video personalizado
python video_processor.py --video mi_video.mp4

# Especificar directorio de salida personalizado
python video_processor.py --output mi_directorio_salida

# Combinación de parámetros
python video_processor.py --video videos/seguridad.mp4 --output resultados/personas
```

### Parámetros de Configuración:

- `--video, -v`: Ruta al video de entrada (default: `input/test_video.mp4`)
- `--output, -o`: Directorio de salida (default: `output/cropped_persons`)

---

## 📁 Estructura de Archivos

```
procesamiento-imagenes-unlu/
├── video_processor.py          # Script principal
├── requirements.txt           # Dependencias Python
├── README.md                 # Esta documentación
├── input/                    # Videos de entrada
│   └── test_video.mp4       # Video de prueba
└── output/                   # Resultados
    └── cropped_persons/      # Imágenes de personas extraídas
        ├── frame_000001_person_1_conf_0.85.jpg
        ├── frame_000001_person_2_conf_0.92.jpg
        └── ...
```

---

## 📊 Salida y Resultados

### Información Durante la Ejecución:

El script proporciona información detallada en tiempo real:
- Progreso del procesamiento (% completado)
- Número de personas detectadas por frame
- Nivel de confianza de cada detección
- Estadísticas finales del procesamiento

### Ejemplo de Salida:
```
🎯 Stage 1: Detección y Extracción de Personas
==================================================
✅ Directorios configurados:
   - Entrada: input/
   - Salida: output/cropped_persons/

🔄 Cargando modelo YOLOv8n...
✅ Modelo YOLOv8n cargado exitosamente
   - Arquitectura: CNN de una sola etapa (single-stage)
   - Dataset de entrenamiento: COCO (80 clases)
   - Clase objetivo: 'person' (ID: 0)

📹 Información del video:
   - Frames totales: 1500
   - FPS: 30.00
   - Duración: 50.00 segundos

🚀 Iniciando procesamiento...

🔍 Frame 30/1500 (2.0%)
   💾 Guardado: frame_000030_person_1_conf_0.87.jpg (conf: 0.87)
   💾 Guardado: frame_000030_person_2_conf_0.92.jpg (conf: 0.92)

==================================================
📊 RESUMEN DEL PROCESAMIENTO
==================================================
✅ Frames procesados: 1500
👥 Personas extraídas: 342
📁 Imágenes guardadas en: output/cropped_persons/

🎯 Stage 1 completado exitosamente!
💡 Las imágenes extraídas están listas para el Stage 2
   (Fine-tuning para detección de armas)
```

---

## 🔬 Consideraciones Técnicas

### Optimizaciones Implementadas:

1. **Umbral de Confianza**: Filtrado a 0.5 para balance precisión/recall
2. **Validación de Coordenadas**: Verificación de límites del frame
3. **Manejo de Memoria**: Liberación de recursos por frame
4. **Nomenclatura Sistemática**: Trazabilidad completa del origen

### Limitaciones Conocidas:

1. **Oclusiones Parciales**: Personas parcialmente ocultas pueden no detectarse
2. **Resolución Mínima**: Personas muy pequeñas (<20px) pueden ignorarse
3. **Condiciones Extremas**: Iluminación muy pobre puede afectar detección

### Recomendaciones para Mejores Resultados:

1. **Calidad de Video**: Usar resolución HD o superior
2. **Iluminación**: Evitar videos con iluminación extremadamente pobre
3. **Estabilidad**: Videos estables reducen falsos positivos por blur

---

## 📚 Referencias y Fuentes

### Documentación Técnica:
- [YOLOv8 Official Documentation](https://docs.ultralytics.com/)
- [COCO Dataset Paper](https://arxiv.org/abs/1405.0312)
- [OpenCV Video Documentation](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)

### Investigación Académica:
- Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- Lin, T.-Y., et al. "Microsoft COCO: Common Objects in Context" (2014)
- Jocher, G., et al. "YOLOv8: A new state-of-the-art computer vision model" (2023)

### Repositorios de Código:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV Python](https://github.com/opencv/opencv-python)

---

## 👨‍💻 Información del Desarrollo

**Proyecto**: Procesamiento de Imágenes - Universidad Nacional de Luján  
**Curso**: Procesamiento Digital de Imágenes  
**Fecha**: Septiembre 2025  
**Tecnologías**: Python, YOLOv8, OpenCV, PyTorch  

**Objetivo Académico**: Demostrar la aplicación práctica de técnicas de Computer Vision en sistemas de detección automatizada, preparando la base para análisis de seguridad más avanzados.

---

*Este documento proporciona la fundamentación completa del Stage 1 del sistema de detección. Para consultas técnicas o mejoras, referirse a la documentación oficial de las librerías utilizadas.*