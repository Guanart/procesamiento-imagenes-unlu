# 🚀 Pipeline Completo de Entrenamiento

Sistema automatizado para entrenar modelos de detección de armas desde cero hasta evaluación final.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Instalación](#instalación)
- [Uso Básico](#uso-básico)
- [Ejemplos Avanzados](#ejemplos-avanzados)
- [Estructura de Archivos](#estructura-de-archivos)
- [Solución de Problemas](#solución-de-problemas)

## 🎯 Descripción

`pipeline_entrenamiento.py` automatiza el flujo completo de entrenamiento:

1. **Split Dataset** → Separa 10% para testing (antes de augmentation)
2. **Data Augmentation** → Genera 2-3 versiones por imagen con transformaciones
3. **Training** → Entrena Faster R-CNN con checkpoints y early stopping
4. **Testing** → Evalúa métricas finales (mAP) en test set no visto

### ✨ Características

- ✅ **Automatización completa**: Un solo comando ejecuta todo el pipeline
- ✅ **Protección contra data leakage**: Test set separado ANTES del augmentation
- ✅ **Fault-tolerant**: Sistema de checkpoints cada N épocas
- ✅ **Resumable**: Reanudar desde cualquier etapa o checkpoint
- ✅ **Métricas profesionales**: mAP, mAP@50, mAP@75, métricas por tamaño
- ✅ **Logs detallados**: JSON con resultados de cada etapa

## 🏗️ Arquitectura del Pipeline

```
dataset/images/           dataset/xmls/
     │                         │
     ├─────────┬───────────────┤
     │         │               │
     ▼         ▼               ▼
┌─────────────────────────────────┐
│  ETAPA 1: Split Dataset         │
│  split_dataset.py               │
│  • Separa 10% para test         │
│  • Seed=42 (reproducible)       │
└─────────────────────────────────┘
     │                         │
     ▼                         ▼
dataset/images/           dataset_testing/
(90% training)            (10% test)
     │                         │
     ▼                         │
┌─────────────────────────────────┐
│  ETAPA 2: Data Augmentation     │
│  augment_dataset.py             │
│  • Flip, rotate, brightness     │
│  • Contrast, saturation, blur   │
│  • Mantiene bounding boxes      │
└─────────────────────────────────┘
     │
     ▼
dataset_augmented/
(180-270% más datos)
     │
     ▼
┌─────────────────────────────────┐
│  ETAPA 3: Training              │
│  train_fasterrcnn_light.py      │
│  • Faster R-CNN + MobileNetV3   │
│  • CLAHE + brightness (--enhance)│
│  • Checkpoints cada 15 épocas   │
│  • Early stopping patience=5    │
└─────────────────────────────────┘
     │
     ▼
results_full/
├── best_model.pth
├── checkpoint_epoch_*.pth
└── training_log.json
     │
     ├─────────────────────────────┐
     │                             │
     ▼                             ▼
┌─────────────────────────────────┐
│  ETAPA 4: Testing               │
│  test_light_model.py            │
│  • Evalúa en test set no visto  │
│  • Calcula mAP completo         │
│  • Guarda imágenes detectadas   │
└─────────────────────────────────┘
     │
     ▼
test_results/
├── test_metrics.json
└── *_detected.jpg
```

## 📦 Instalación

### Requisitos previos

```bash
# Python 3.8+
python3 --version

# PyTorch con CUDA (recomendado) o CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Dependencias adicionales
pip install opencv-python pillow tqdm torchmetrics matplotlib psutil
```

### Verificar instalación

```bash
cd weapons_detector2
python3 pipeline_entrenamiento.py --help
```

## 🚀 Uso Básico

### Entrenamiento completo (configuración por defecto)

```bash
cd weapons_detector2
python3 pipeline_entrenamiento.py
```

**Esto ejecutará:**
- Split: 10% → test, 90% → training
- Augmentation: 2 versiones por imagen
- Training: 100 épocas, batch_size=8
- Testing: Evalúa en test set

**Duración estimada:** 2-4 horas (depende de GPU)

### Entrenamiento largo (1000 épocas)

```bash
python3 pipeline_entrenamiento.py \
  --epochs 1000 \
  --num-augmentations 3 \
  --save-every 20 \
  --patience 10 \
  --enhance \
  --amp
```

**Características:**
- 1000 épocas con checkpoints cada 20
- 3 versiones aumentadas por imagen (más datos)
- Early stopping si no mejora en 10 épocas
- Image enhancement (CLAHE + brightness)
- Automatic Mixed Precision (más rápido)

## 🔧 Ejemplos Avanzados

### 1. Reanudar entrenamiento interrumpido

```bash
# El split y augmentation ya están hechos, solo entrenar y testear
python3 pipeline_entrenamiento.py \
  --skip-stages split augment \
  --resume results_full/checkpoint_epoch_45.pth \
  --epochs 1000
```

### 2. Solo testing (modelo ya entrenado)

```bash
python3 pipeline_entrenamiento.py \
  --skip-stages split augment train \
  --output-dir results_full
```

### 3. Experimentar con hiperparámetros

```bash
python3 pipeline_entrenamiento.py \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --num-augmentations 1 \
  --output-dir results_experiment
```

### 4. Training rápido para debugging

```bash
python3 pipeline_entrenamiento.py \
  --epochs 5 \
  --batch-size 4 \
  --test-split 0.2 \
  --num-augmentations 1 \
  --output-dir results_debug
```

### 5. Fine-tuning desde modelo existente

```bash
python3 pipeline_entrenamiento.py \
  --skip-stages split augment \
  --resume results_full/best_model.pth \
  --epochs 200 \
  --learning-rate 1e-5 \
  --output-dir results_finetune
```

## 📁 Estructura de Archivos

### Antes del pipeline

```
weapons_detector2/
├── dataset/
│   ├── images/              # 100 imágenes originales
│   │   ├── img001.jpg
│   │   ├── img002.png
│   │   └── ...
│   └── xmls/                # 100 anotaciones
│       ├── img001.xml
│       ├── img002.xml
│       └── ...
├── pipeline_entrenamiento.py
├── split_dataset.py
├── augment_dataset.py
├── train_fasterrcnn_light.py
├── test_light_model.py
└── image_enhancer.py
```

### Después del pipeline

```
weapons_detector2/
├── dataset/
│   ├── images/              # 90 imágenes (10% movidas a test)
│   └── xmls/                # 90 anotaciones
│
├── dataset_testing/         # ← NUEVO: Test set (10%)
│   ├── images/              # 10 imágenes reservadas
│   └── xmls/                # 10 anotaciones
│
├── dataset_augmented/       # ← NUEVO: Training aumentado
│   ├── images/              # 90 × 3 = 270 imágenes
│   │   ├── img001.jpg
│   │   ├── img001_aug_1.jpg
│   │   ├── img001_aug_2.jpg
│   │   └── ...
│   └── xmls/                # 270 XMLs con boxes ajustados
│
├── results_full/            # ← NUEVO: Modelo entrenado
│   ├── best_model.pth       # Mejor modelo por mAP
│   ├── checkpoint_epoch_15.pth
│   ├── checkpoint_epoch_30.pth
│   ├── checkpoint_epoch_45.pth
│   ├── training_log.json    # Historial completo
│   ├── training_history.png # Gráficos
│   ├── classes.json
│   └── pipeline_results.json # ← Resumen del pipeline
│
└── test_results/            # ← NUEVO: Evaluación final
    ├── test_metrics.json    # mAP, mAP@50, mAP@75, etc.
    ├── img091_detected.jpg  # Visualizaciones
    ├── img092_detected.jpg
    └── ...
```

## 📊 Interpretar Resultados

### 1. Ver resumen del pipeline

```bash
cat results_full/pipeline_results.json
```

```json
{
  "pipeline_start": "2025-11-20T10:30:00",
  "pipeline_end": "2025-11-20T13:45:30",
  "total_duration_sec": 11730,
  "success": true,
  "stages": {
    "1. Split Dataset": {
      "success": true,
      "duration_sec": 2.5
    },
    "2. Data Augmentation": {
      "success": true,
      "duration_sec": 180.3
    },
    "3. Train Model": {
      "success": true,
      "duration_sec": 11400.0
    },
    "4. Test Model": {
      "success": true,
      "duration_sec": 45.8
    }
  }
}
```

### 2. Ver métricas de test

```bash
cat test_results/test_metrics.json
```

```json
{
  "model_path": "results_full/best_model.pth",
  "test_images": 10,
  "processed": 10,
  "failed": 0,
  "confidence_threshold": 0.5,
  "total_ground_truth": 25,
  "total_detections": 23,
  "metrics": {
    "map": 0.7856,
    "map_50": 0.9234,
    "map_75": 0.8012,
    "map_small": 0.6543,
    "map_medium": 0.8234,
    "map_large": 0.9012
  }
}
```

**Interpretación:**
- ✅ **mAP = 0.7856**: Excelente (>0.7 es muy bueno)
- ✅ **mAP@50 = 0.9234**: Detecciones bien localizadas
- ⚠️ **mAP small = 0.6543**: Objetos pequeños más difíciles
- ✅ **23/25 detecciones**: 92% de recall

### 3. Ver historial de entrenamiento

```bash
python3 << EOF
import json
import matplotlib.pyplot as plt

with open('results_full/training_log.json') as f:
    history = json.load(f)

epochs = [h['epoch'] for h in history]
val_loss = [h['val_loss'] for h in history]
map_values = [h['map'] for h in history]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, val_loss)
plt.title('Validation Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(epochs, map_values)
plt.title('Mean Average Precision')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('results_analysis.png')
print("Gráficos guardados en results_analysis.png")
EOF
```

## 🛠️ Solución de Problemas

### Problema 1: OOM (Out of Memory)

**Síntoma:**
```
RuntimeError: CUDA out of memory
```

**Solución:**
```bash
# Reducir batch size
python3 pipeline_entrenamiento.py --batch-size 4

# O desactivar AMP si causa problemas
python3 pipeline_entrenamiento.py --batch-size 4
# (sin --amp)
```

### Problema 2: Pipeline se detiene en augmentation

**Síntoma:**
```
❌ Error: Imagen no encontrada para: img042.xml
```

**Solución:**
```bash
# Verificar que todas las imágenes tengan XML correspondiente
cd dataset
ls images/ | wc -l
ls xmls/ | wc -l
# Los números deben coincidir

# Encontrar XMLs sin imagen
for xml in xmls/*.xml; do
    base=$(basename "$xml" .xml)
    if [ ! -f "images/$base.jpg" ] && [ ! -f "images/$base.png" ]; then
        echo "Sin imagen: $xml"
    fi
done
```

### Problema 3: Test metrics mAP = 0

**Síntoma:**
```json
"map": 0.0000
```

**Causas posibles:**
1. **Modelo no entrenó bien**: Revisar `training_history.png`
2. **Confidence threshold muy alto**: Probar `--confidence 0.3`
3. **Test set muy pequeño**: Usar `--test-split 0.2` (20% test)

**Solución:**
```bash
# Re-testear con confidence más bajo
python3 test_light_model.py \
  --model results_full/best_model.pth \
  --confidence 0.3
```

### Problema 4: Quiero cambiar el test split después de empezar

**Solución:** Necesitas volver a empezar desde cero (el split ya se hizo)

```bash
# 1. Restaurar dataset original
mv dataset_testing/images/* dataset/images/
mv dataset_testing/xmls/* dataset/xmls/
rm -r dataset_testing dataset_augmented results_full test_results

# 2. Re-ejecutar con nuevo split
python3 pipeline_entrenamiento.py --test-split 0.15
```

### Problema 5: El entrenamiento se estancó

**Síntoma:**
```
Val Loss no mejoró (5/5)
⏹️  Early stopping activado
```

**Esto es normal!** El early stopping detectó overfitting.

**Revisar:**
```bash
# Ver última época antes de detener
tail -20 results_full/training_log.json
```

Si mAP es bueno (>0.6), está bien. Si no:
- Aumentar `--patience 10` (más tolerante)
- Aumentar `--num-augmentations 3` (más datos)
- Reducir `--learning-rate 5e-5` (aprendizaje más lento)

## 📚 Referencias

- **TRAINING_GUIDE.md**: Guía detallada de entrenamiento manual
- **train_fasterrcnn_light.py**: Script de entrenamiento (documentación interna)
- **test_light_model.py**: Script de evaluación (documentación interna)

## 🎓 Preguntas Frecuentes

**P: ¿Cuánto tarda el pipeline completo?**  
R: 2-4 horas para 100 épocas, 10-20 horas para 1000 épocas (con GPU)

**P: ¿Puedo usar CPU en vez de GPU?**  
R: Sí, pero será 10-20x más lento. Usa `--epochs 10` para probar.

**P: ¿Cuánto espacio en disco necesito?**  
R: ~5GB para dataset aumentado, ~2GB para checkpoints, ~500MB para results

**P: ¿Qué mAP es "bueno"?**  
R: 
- mAP > 0.7 = Excelente
- mAP 0.5-0.7 = Bueno
- mAP 0.3-0.5 = Aceptable (necesita más entrenamiento)
- mAP < 0.3 = Problema (revisar datos/hiperparámetros)

**P: ¿Puedo interrumpir el pipeline y continuar después?**  
R: Sí! Usa `--skip-stages` y `--resume` (ver ejemplos arriba)

---

**🎉 ¡Listo para entrenar!**

```bash
cd weapons_detector2
python3 pipeline_entrenamiento.py --epochs 100 --enhance --amp
```
