# 🔫 Detección de Armas en Tiempo Real - UNLu

Proyecto de detección de armas (pistola/cuchillo) en personas usando Faster R-CNN optimizado para AMD ROCm (Radeon 780M iGPU).

---

## 🚀 Inicio Rápido

### 1. Verificar GPU AMD
```bash
python check_gpu.py
```

### 2. Entrenar detector
```bash
python weapons_detector/train_fasterrcnn.py \
  --images-dir dataset/images \
  --xml-dir dataset/xmls \
  --output-dir results_frcnn \
  --epochs 20 \
  --batch-size 4 \
  --amp
```

### 3. Detección en tiempo real
```bash
python weapons_detector/real_time_weapon_detector.py \
  --model-path results_frcnn/best_model.pth \
  --classes-path results_frcnn/classes.json
```

---

## 📋 Requisitos
- Python 3.8+
- PyTorch con ROCm (AMD GPU) o CUDA
- WSL2 en Windows 11 (para AMD)
- 20GB RAM
- Dataset con imágenes + XML (formato Pascal VOC)

## 🔧 Instalación

### PyTorch ROCm (AMD Radeon 780M)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

### Dependencias
```bash
pip install opencv-python scikit-learn matplotlib tqdm ultralytics
```

---

## 📂 Estructura del Dataset

```
dataset/
├── images/              # Todas las imágenes mezcladas
│   ├── armas (1).jpg
│   ├── knife_001.jpg
│   ├── pistol_045.jpg
│   └── ...
└── xmls/                # Archivos XML (uno por imagen)
    ├── armas (1).xml
    ├── knife_001.xml
    ├── pistol_045.xml
    └── ...
```

### Formato XML (Pascal VOC)
```xml
<annotation>
  <filename>armas (1).jpg</filename>
  <size>
    <width>240</width>
    <height>145</height>
  </size>
  <object>
    <name>pistol</name>  <!-- o 'knife' -->
    <bndbox>
      <xmin>3</xmin>
      <ymin>1</ymin>
      <xmax>128</xmax>
      <ymax>100</ymax>
    </bndbox>
  </object>
</annotation>
```

**Notas:**
- Campo `<path>` se ignora (puede estar incorrecto)
- Clases: `pistol` y `knife`
- Múltiples objetos por imagen permitidos

---

## 🎯 Entrenamiento

### Comando recomendado (AMD 780M)
```bash
python weapons_detector/train_fasterrcnn.py \
  --images-dir dataset/images \
  --xml-dir dataset/xmls \
  --output-dir results_frcnn \
  --epochs 20 \
  --batch-size 4 \
  --lr 0.001 \
  --amp
```

### Parámetros
| Parámetro | Descripción | Valor recomendado |
|-----------|-------------|-------------------|
| `--epochs` | Número de épocas | 15-25 |
| `--batch-size` | Tamaño del batch | 4 (2 si OOM) |
| `--lr` | Learning rate | 0.001 |
| `--amp` | Mixed precision | ✅ Sí |

### Salida
```
results_frcnn/
├── best_model.pth       # Modelo entrenado
├── classes.json         # Mapeo clases
└── training_log.json    # Historial
```

---

## 🎬 Detección en Tiempo Real

### Webcam
```bash
python weapons_detector/real_time_weapon_detector.py \
  --model-path results_frcnn/best_model.pth \
  --classes-path results_frcnn/classes.json
```

### Video
```bash
python weapons_detector/real_time_weapon_detector.py \
  --model-path results_frcnn/best_model.pth \
  --classes-path results_frcnn/classes.json \
  --video input.mp4
```

**Controles:** Presiona `q` para salir

---

## ⚙️ Optimizaciones AMD Radeon 780M

### Configuración WSL2 (.wslconfig)
```ini
[wsl2]
memory=20GB
processors=8
swap=8GB
```

### Variables de entorno (automáticas)
```bash
PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
```

### Consejos
- ✅ Siempre usar `--amp` (2-3x más rápido)
- ✅ Dataset en disco NVMe (no /mnt/c)
- ⚠️ Reducir batch-size si "OOM error"

---

## 📊 Métricas Esperadas (780M)

| Métrica | Valor |
|---------|-------|
| Velocidad entrenamiento | 15-20 imgs/seg |
| Inferencia webcam | 10-15 FPS |
| Memoria GPU | 2-4 GB |

---

## 🐛 Solución de Problemas

### GPU no detectada
```bash
python check_gpu.py
# Si False, reinstalar PyTorch ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

### CUDA out of memory
- `--batch-size 2`
- Cerrar otras apps GPU

### Imágenes no encontradas
- Verificar `<filename>` en XML coincide con archivo
- Campo `<path>` se ignora automáticamente

---

## 📁 Archivos del Proyecto

### ✅ Principales (usar)
- [`weapons_detector/train_fasterrcnn.py`](weapons_detector/train_fasterrcnn.py) - Entrenamiento
- [`weapons_detector/real_time_weapon_detector.py`](weapons_detector/real_time_weapon_detector.py) - Detección
- [`check_gpu.py`](check_gpu.py) - Verificación GPU

### ℹ️ Auxiliares
- `weapons_classifier/` - Clasificador simple (NO usar para detección)
- `dataset_tools/` - Scripts conversión formatos
- `person_extraction/` - Stage 1 (YOLO personas)

---

**Uso:**
```bash
cd person_extraction
python video_processor.py
python image_enhancer.py
```

**Resultado:** 270 personas extraídas y mejoradas

### 3. flask_analyzer
Servidor web para análisis de imágenes (histogramas RGB, estadísticas).

**Uso:**
```bash
cd flask_analyzer
python app.py
```

Acceder a: `http://localhost:5000`

### 4. weapons_classifier
Clasificador de armas (pistolas vs cuchillos) usando PyTorch.
**Uso:**
```bash


## 🎓 Flujo de Trabajo Completo

```
1. Verificar GPU       → python check_gpu.py
2. Preparar dataset    → Organizar images/ y xmls/
3. Entrenar modelo     → python weapons_detector/train_fasterrcnn.py ...
4. Detectar tiempo real → python weapons_detector/real_time_weapon_detector.py ...
```

---

## 📝 Notas Importantes

- ✅ **Usar Faster R-CNN** (train_fasterrcnn.py) para detección
- ❌ **NO usar train_weapons_classifier.py** (solo clasificación básica)
- ✅ Dataset debe tener personas sosteniendo armas (contexto real)
- ❌ No usar imágenes fondos lisos (no detecta en personas)
- ✅ Mixed Precision (--amp) esencial para AMD

---

## 📚 Referencias

- [Faster R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

---

**Proyecto académico** - Universidad Nacional de Luján  
Procesamiento de Imágenes - 2025

