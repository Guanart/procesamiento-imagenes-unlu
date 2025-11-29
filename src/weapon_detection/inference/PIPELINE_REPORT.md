# Weapon Detection Pipeline Overview

Este informe resume el flujo extremo a extremo que implementamos entre los módulos `src/weapon_detection/training`, `src/person_extraction` y `src/weapon_detection/inference`.

## Componentes Principales
- **Detección/recorte de personas (`person_extraction`)**: usa YOLOv8n (`yolov8n.pt`) para localizar personas en video o imagen completa. Cada bounding box válido (confidence ≥ 0.5) se recorta y conserva la posición original.
- **Mejora de imagen (`src/person_extraction/image_enhancer.py`)**: a cada recorte se le aplican CLAHE + boost del canal V para mejorar contraste/brillo. También se reescala si es muy pequeño.
- **Detector de armas (`src/weapon_detection/training/train_fasterrcnn_light.py`)**: Faster R-CNN con backbone MobileNetV3 Large FPN entrenado para dos clases (`knife`, `pistol`). Se usa `torchvision` con AMP y AdamW.
- **Orquestador (`src/weapon_detection/inference/detector_pipeline.py`)**: coordina las etapas anteriores para cualquier imagen o video, pinta anotaciones y puede exportar video con H.264.

## Entrenamiento del detector de armas
- **Dataset**: 9,967 anotaciones para entrenamiento + 1,759 para validación (formatos Pascal VOC). Cada anotación incluye cajas de pistolas o cuchillos.
- **Configuración**: batch size 8, imágenes redimensionadas a 320×320, augmentación previa y `ImageEnhancer` opcional. Entrenamiento típico en GPU (Tesla T4) con AMP activado.
- **Métricas**: en Colab se alcanzó mAP 0.616 @IoU0.5 en la época 2 (checkpoint `results_standard/best_model.pth`). Training Loss ≈ 0.59, Val Loss ≈ 0.58.
- **Reanudación**: `train_fasterrcnn_light.py --resume <checkpoint>` retoma desde la época guardada, preservando optimizer, historial y mejor mAP.

## Flujo del Pipeline (`pipeline.py`)
1. **Carga de modelos**: YOLO para personas (CPU/GPU automático) y Faster R-CNN MobileNetV3 para armas.
2. **Detección de personas**: por frame, YOLO produce cajas. Se filtra por `min_person_conf` y cada recorte se clampa al frame original.
3. **Mejora de recortes**: `enhance_image` aplica CLAHE + ajuste HSV para robustecer la señal antes de buscar armas.
4. **Detección y clasificación de armas**: el recorte mejorado se pasa a Faster R-CNN. Cada predicción con score ≥ `confidence_threshold` devuelve clase (`knife`/`pistol`), score y caja relativa.
5. **Reconstrucción en frame original**: las cajas de armas se trasladan usando las coordenadas de la persona. Se dibuja la caja amarilla para la persona y roja para cada arma detectada, junto con texto `PERSON WITH WEAPON`.
6. **Procesamiento de video**: permite `frame_skip`, registro de progreso y escritura H.264 (ffmpeg). Cuenta detecciones totales y soporta callback externo.

## Cómo ejecutar
```bash
# Entrenar / reanudar detector ligero
default_cmd="python3 src/weapon_detection/training/train_fasterrcnn_light.py \
  --images-dir dataset_augmented/images \
  --xml-dir dataset_augmented/xmls \
  --output-dir results_standard \
  --epochs 50 --batch-size 8 --lr 1e-4 \
  --amp --enhance --save-every 3 --patience 5"

# Reanudar desde el mejor modelo
eval "$default_cmd --resume results_standard/best_model.pth"

# Ejecutar pipeline completo sobre un video
python3 src/weapon_detection/inference/detector_pipeline.py \
  --input sample_video.mp4 \
  --output detections.mp4 \
  --confidence 0.5 --frame-skip 2
```

## Resultado final
- **Entrada**: video/imagen completa.
- **Salida**: frame/video anotado con cajas amarillas (personas) y rojas (armas) + JSON en memoria con ubicación y tipo de arma.
- **Uso típico**: monitoreo de cámaras, evidencia visual y etapas posteriores de clasificación o alerta.

Con este resumen podés limitarte a mantener solo los scripts clave (`train_fasterrcnn_light.py`, `pipeline.py`, `detector_pipeline.py`, utilidades de imagen) sin depender de la documentación extensa que eliminamos.
