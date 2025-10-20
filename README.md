# 🎯 Procesamiento de Imágenes - Universidad Nacional de Luján

**Sistema completo de procesamiento de imágenes y detección de person### 1️⃣ Sistema Flask - Análisis de Imágenes 🌐
```bash
# Ejecutar servidor web
python app.py

# Abrir navegador en: http://localhost:5000
# Drag & drop de cualquier imagen (JPG, PNG, BMP)
```
**¿Qué hace?**: Analiza canales RGB, genera histogramas, extrae metadatos completos

### 2️⃣ Sistema YOLOv8 - Detección de Personas 🤖

Este proyecto implementa **dos sistemas integrados** para análisis de imá## ⚠️ Limitaciones y Consideraciones

### **YOLOv8 - Detección de Personas**:
- ❌ No detecta siluetas artificiales (solo personas reales)
- 📐 Precisión depende de calidad del video (mín. 480p)
- 🌙 Funciona mal con iluminación extremadamente pobre
- ⏱️ Videos largos requieren tiempo considerable

### **Flask - Análisis de Imágenes**:
- 🔒 Sin autenticación (solo desarrollo local)
- 💾 Sin persistencia de resultados entre sesiones
- 📊 Limitado a una imagen por análisis

### **Recomendaciones de Calidad**:
- ✅ **Videos HD+** (1280x720 o superior)
- ✅ **Buena iluminación** y cámara estable
- ✅ **Personas visibles** >50% del cuerpo
- ✅ **30 FPS** para mejor tracking y detección inteligente:

## � Características Principales

### 🌐 Sistema Flask - Análisis de Imágenes
- ✅ **Análisis RGB completo** por canales de color
- ✅ **Histogramas interactivos** generados server-side
- ✅ **Metadatos detallados** (resolución, formato, tamaño)
- ✅ **Estadísticas avanzadas** (min, max, promedio, moda, desviación)
- ✅ **Interfaz moderna** con drag & drop y Tailwind CSS

### 🤖 Sistema YOLOv8 - Detección de Personas
- ✅ **Detección automática** de personas en videos (IA pre-entrenada)
- ✅ **Extracción inteligente** de imágenes por timestamp
- ✅ **Modelo YOLOv8n** optimizado para tiempo real
- ✅ **Dataset COCO** para máxima precisión en personas
- ✅ **Pipeline de mejora** con interpolación spline y realce de calidad
- ✅ **Preparación Stage 2** para detección de armas en seguridad

### 🔄 Sistema Data Augmentation - Aumento de Dataset
- ✅ **Aumentación simple y controlada** para armas (cuchillos, pistolas)
- ✅ **Cuadruplica el dataset** (Original + 3 transformaciones: flips y rotación)
- ✅ **Orientado a objetos** para un código limpio y encapsulado
- ✅ **Conserva la estructura de clases** (subdirectorios)
- ✅ **Limpia el directorio de salida** antes de cada ejecución

**📊 Estado**: ✅ Completamente funcional y probado (Octubre 2025)

## 🎯 Casos de Uso y Dataset

### 🔍 Aplicaciones del Sistema

#### **Análisis de Imágenes (Flask)**:
- 📚 **Educativo**: Enseñanza de conceptos RGB y procesamiento digital
- 🔬 **Investigación**: Análisis de propiedades colorimétricas
- 📊 **Técnico**: Extracción de metadatos y estadísticas de imagen

#### **Detección de Personas (YOLOv8)**:
- 🛡️ **Seguridad**: Análisis de videos de vigilancia
- 📹 **Investigación**: Estudio de comportamiento humano
- 🎯 **Dataset Generation**: Preparación para Stage 2 (detección de armas)

#### **Data Augmentation (Armas)**:
- 🔪 **Entrenamiento IA**: Cuadruplicar dataset de cuchillos y pistolas
- 🎯 **Balanceo de clases**: Igualar cantidad de muestras por tipo
- 📊 **Mejora de precisión**: Más datos = mejor modelo
- 🔄 **Automatización**: Generar 4 imágenes por cada original

### 📁 Formatos y Videos Soportados

#### **Imágenes** (Sistema Flask):
- ✅ JPG, JPEG, PNG, BMP, TIFF
- 📏 Cualquier resolución
- 💾 Análisis completo de metadatos

#### **Videos** (Sistema YOLOv8):
- ✅ MP4, AVI, MOV, MKV, WMV
- 📺 Resolución mínima: 480p (recomendado HD+)
- ⚡ FPS: 15+ (recomendado 30 FPS)

**📹 Videos de ejemplo**: [Pexels - People Walking](https://www.pexels.com/search/videos/people%20walking/)

## � Tecnología IA: YOLOv8 + COCO Dataset

### **¿Por qué YOLOv8?**
- 🚀 **Velocidad**: Procesamiento en tiempo real (single-stage CNN)
- 🎯 **Precisión**: >90% accuracy en detección de personas
- ⚡ **Eficiencia**: Versión nano optimizada para recursos limitados
- 🔧 **Pre-entrenado**: Modelo COCO listo para usar

### **COCO vs ImageNet**: ¿Por qué COCO?
- ✅ **COCO**: Específico para **detección de objetos** con coordenadas espaciales
- ✅ **Clase 'persona'**: Altamente optimizada (ID: 0 en 80 clases)
- ❌ **ImageNet**: Solo clasificación, sin localización espacial

## 📋 Funcionalidades Actuales

La aplicación actualmente implementa:

### Análisis de Metadatos
- Formato de imagen (JPG, PNG, etc.)
- Dimensiones (ancho x alto en píxeles)
- Tamaño en disco (bytes y MB)
- Modo de color y número de canales

### Análisis por Canal RGB
Para cada canal de color (Rojo, Verde, Azul):
- 📊 Histograma de distribución de píxeles
- 📉 Valor mínimo
- 📈 Valor máximo
- ➗ Promedio (media aritmética)
- 📊 Desviación estándar
- 🎯 Moda (valor más frecuente)

### Interfaz de Usuario
- Diseño moderno y responsive con Tailwind CSS
- Carga de imágenes mediante drag & drop o selección
- Visualización organizada de resultados por canal
- Tabla comparativa de estadísticas

## 🚀 Inicio Rápido

### 0️⃣ Instalar Dependencias
```bash
# Clonar repositorio
git clone https://github.com/Guanart/procesamiento-imagenes-unlu.git
cd procesamiento-imagenes-unlu

# Instalar librerías (incluyendo scipy y tqdm para mejora de imágenes)
pip install -r requirements.txt
```

### 1️⃣ Sistema Flask - Análisis de Imágenes 🌐
```bash
# Ejecutar servidor web
python app.py

# Abrir navegador en: http://localhost:5000
# Drag & drop de cualquier imagen (JPG, PNG, BMP)
```
**¿Qué hace?**: Analiza canales RGB, genera histogramas, extrae metadatos completos

### 2️⃣ Sistema YOLOv8 - Detección de Personas 🤖
```bash
# Colocar video en carpeta input/
cp tu_video.mp4 input/

# Ejecutar detección IA
python video_processor.py --video input/tu_video.mp4

# Ver personas extraídas en: output/cropped_persons/
```
**¿Qué hace?**: Detecta personas automáticamente y extrae sus imágenes frame por frame

### 3️⃣ Sistema de Mejora de Calidad - Pipeline de Imágenes 🎨
```bash
# Mejorar calidad de personas extraídas
# Aplica interpolación spline, reducción de ruido, realce de nitidez
python image_enhancer.py

# Las imágenes mejoradas se guardan en: output/enhanced_persons/

# Personalizar tamaños mínimos
python image_enhancer.py --min-height 250 --min-width 120
```
**¿Qué hace?**: Mejora la calidad de las imágenes de personas usando interpolación spline cúbica (redimensionamiento a mínimo 200x100 px), reducción de ruido, realce de nitidez, mejora de contraste (CLAHE) y realce de bordes. Esto prepara las imágenes para el Stage 2 (detección de armas).

### 4️⃣ Sistema Data Augmentation - Aumento de Dataset 🔄
```bash
# Organizar dataset por clases (si no está hecho)
mkdir -p dataset/original/knife dataset/original/pistol
# cp cuchillo*.jpg dataset/original/knife/
# cp pistol*.jpg dataset/original/pistol/

# Ejecutar aumentación simple
# Esto leerá de 'dataset/original' y escribirá en 'dataset/augmented'
python simple_augmenter.py

# Para especificar directorios
python simple_augmenter.py --input /ruta/a/tus/imagenes --output /ruta/de/salida
```
**¿Qué hace?**: Cuadruplica tu dataset (copia el original y añade 3 transformaciones: flip horizontal, vertical y rotación de 90º).

### 5️⃣ Generación de Informe - Análisis del Dataset 📊
```bash
# Generar informe completo del dataset
python generate_report.py

# El informe se guarda en: INFORME_DATASET.md y INFORME_DATASET.json

# Personalizar directorios
python generate_report.py \
  --weapons-dir dataset/augmented \
  --persons-dir output/cropped_persons \
  --enhanced-dir output/enhanced_persons \
  --output MI_INFORME.md
```
**¿Qué hace?**: Genera un informe completo que incluye:
- Balanceo de clases (pistolas vs cuchillos)
- Normalización de tamaños de imágenes
- Verificación de transformaciones aplicadas
- Estado del pipeline de mejora de personas
- Recomendaciones para el siguiente paso

### 🐳 Alternativa Docker
```bash
docker-compose up --build
# Flask disponible en: http://localhost:5000
```

## � Ejemplos de Resultados

### 🌐 Sistema Flask - Análisis RGB
```
📸 Imagen: foto.jpg (1920x1080, 2.1MB)
📊 Canal Rojo:   Min: 12  | Max: 248 | Promedio: 128.5 | Moda: 156
📊 Canal Verde:  Min: 8   | Max: 255 | Promedio: 142.1 | Moda: 178  
📊 Canal Azul:   Min: 15  | Max: 240 | Promedio: 98.7  | Moda: 92
📈 Histogramas generados server-side (evita crashes del navegador)
```

### 🤖 Sistema YOLOv8 - Detección
```
🎬 Video: seguridad.mp4 (1500 frames, 30 FPS, 50s)
🔍 Procesando frame 450/1500 (30.0%)
👥 Persona detectada: confidence 0.87 → frame_000450_person_1_conf_0.87.jpg
👥 Persona detectada: confidence 0.92 → frame_000450_person_2_conf_0.92.jpg

📊 RESULTADO FINAL:
✅ Frames procesados: 1500
👥 Personas extraídas: 342 imágenes
📁 Guardado en: output/cropped_persons/
```

## 🔧 Configuración Avanzada

### **Parámetros YOLOv8**:
```bash
python video_processor.py \
  --video input/mi_video.mp4 \
  --confidence 0.7 \              # Umbral de confianza (0.5 default)
  --output resultados/personas    # Directorio personalizado
```

### **Optimización de Rendimiento**:
- 🖥️ **GPU**: CUDA automática si disponible
- 💾 **RAM**: 4GB mínimo, 8GB recomendado  
- ⚡ **Velocidad**: ~10-15 FPS (HD) con GPU moderna

## 🛠️ Desarrollo Futuro - Roadmap

### **Stage 2** - Detección de Armas 🎯
- 🔫 Fine-tuning YOLOv8 para detectar armas en personas
- 🎓 Transfer learning desde detección de personas
- 📊 Uso de dataset: [DASCI Detección de Armas](https://dasci.es/opendata/deteccion-de-armas-open-data/)
- 🔄 **Data augmentation implementado**: Expandir dataset a 1500-2000 imágenes/clase

### **Mejoras Técnicas** ⚡
- 🌐 Interfaz web unificada (Flask + YOLOv8)
- 🔄 Procesamiento batch de múltiples videos
- 📡 API REST para integración con otros sistemas
- 📈 Dashboard de analytics y métricas

## 📁 Estructura del Proyecto

```
procesamiento-imagenes-unlu/
├── 🌐 SISTEMA FLASK
│   ├── app.py                    # Servidor Flask
│   ├── templates/index.html      # Interfaz web
│   ├── Dockerfile               # Container config
│   └── docker-compose.yml       # Orquestación
│
├── 🤖 SISTEMA YOLO
│   ├── video_processor.py       # Detección YOLOv8
│   ├── README_DETECTION.md      # Documentación técnica
│   └── yolov8n.pt              # Modelo pre-entrenado
│
├── 📁 DATOS
│   ├── input/                   # Videos de entrada
│   ├── output/                  # Resultados
│   └── uploads/                 # Archivos temporales
│
└── 📚 DOCUMENTACIÓN
    ├── README.md                # Este archivo
    ├── ESTADO_PROYECTO.md       # Estado completo
    └── requirements.txt         # Dependencias
```

## 🔧 Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Procesamiento**: Pillow, NumPy
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Containerización**: Docker, Docker Compose

## 📝 Notas Técnicas

> **Importante**: La resolución radiométrica (profundidad de bits por píxel), resolución espectral (número de bandas espectrales) y otros metadatos específicos no están disponibles en formatos de imagen comunes como JPG o PNG, ya que estos formatos están optimizados para visualización y no conservan información técnica detallada de sensores remotos.

## � Documentación Adicional

- 📖 **[README_DETECTION.md](README_DETECTION.md)** - Documentación técnica completa del sistema YOLOv8
- 📊 **[ESTADO_PROYECTO.md](ESTADO_PROYECTO.md)** - Estado actual y componentes instalados
- 📹 **[input/README.md](input/README.md)** - Guía para videos de entrada

## 🔧 Stack Tecnológico Completo

### **Backend & IA**:
- 🐍 **Python 3.12.3** - Lenguaje principal
- 🌐 **Flask 3.1.2** - Framework web
- 🤖 **YOLOv8n (Ultralytics 8.3.203)** - Detección IA
- 🖼️ **Pillow 11.3.0** - Procesamiento de imágenes
- 📊 **NumPy 2.2.6** - Operaciones numéricas
- 📈 **Matplotlib 3.10.6** - Generación de gráficos
- 🎥 **OpenCV 4.12.0.88** - Procesamiento de video
- 🧠 **PyTorch 2.8.0** - Deep learning backend

### **Frontend & Deploy**:
- 🎨 **HTML5 + Tailwind CSS** - Interfaz moderna
- ⚡ **JavaScript** - Interactividad
- 🐳 **Docker + Docker Compose** - Containerización
- 🔧 **Virtual Environment** - Aislamiento de dependencias

## 🎓 Información Académica

**🏫 Universidad Nacional de Luján**  
**📚 Asignatura**: Procesamiento Digital de Imágenes  
**📅 Período**: Octubre 2025  
**👨‍🎓 Repositorio**: [GitHub - procesamiento-imagenes-unlu](https://github.com/Guanart/procesamiento-imagenes-unlu)

### 📞 Soporte y Contribuciones
- 🐛 **Issues**: [GitHub Issues](https://github.com/Guanart/procesamiento-imagenes-unlu/issues)
- 📖 **Documentación**: Este README (consolidado)
- 🔄 **Updates**: Branch `main` contiene la versión estable

---

## 📈 Estado del Proyecto

| Componente | Estado | Versión | Última Prueba |
|------------|--------|---------|---------------|
| 🌐 **Flask App** | ✅ Funcional | 1.0.0 | Oct 2025 |
| 🤖 **YOLOv8 Detection** | ✅ Funcional | 1.0.0 | Oct 2025 |
| 🐳 **Docker Setup** | ✅ Funcional | 1.0.0 | Oct 2025 |
| 📦 **Dependencies** | ✅ Instaladas | Latest | Oct 2025 |

**🚀 ¡Sistema 100% operativo y listo para usar!**

---
*Desarrollado con ❤️ para educación en procesamiento de imágenes y computer vision*
