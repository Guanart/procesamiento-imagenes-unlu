# Sistema de Detección de Armas

Repositorio académico (UNLu) para detectar armas blancas y de fuego en personas. El proyecto combina entrenamiento de Faster R-CNN, un pipeline de inferencia completo y dos aplicaciones Flask desplegables con Docker.

## Componentes principales

1. **Entrenamiento** (`src/weapon_detection/training/`): scripts para dividir el dataset, aumentarlo, entrenar Faster R-CNN y validar métricas.
2. **Inferencia** (`src/weapon_detection/inference/detector_pipeline.py`): pipeline extremo a extremo que recorta personas con YOLOv8, mejora cada recorte y detecta armas.
3. **Aplicaciones web** (`apps/`):
   - `image_lab/`: laboratorio visual para analizar histograma y metadatos de imágenes.
   - `weapon_monitor/`: dashboard drag & drop con inferencia de imágenes, videos y webcam + API REST.

## Requisitos y entornos

Las dependencias están modularizadas en `requirements/` para instalar solo lo necesario:

| Archivo | Contenido |
| --- | --- |
| `requirements/base.txt` | Inferencia, CLI y utilidades compartidas |
| `requirements/training.txt` | Paquetes extra para entrenamiento (`-r base.txt`) |
| `requirements/apps/image_lab.txt` | Dependencias específicas de Image Lab |
| `requirements/apps/weapon_monitor.txt` | Flask + librerías de Weapon Monitor (`-r ../base.txt`) |

### Instalación rápida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt   # o el archivo que necesites
```

Para trabajar solo con las apps web puedes instalar `requirements/apps/<app>.txt` dentro del mismo entorno.

## Estructura

```
.
├── apps/
│   ├── image_lab/
│   │   ├── app.py
│   │   ├── templates/
│   │   └── Dockerfile
│   └── weapon_monitor/
│       ├── app.py
│       ├── templates/
│       └── Dockerfile
├── docs/
│   ├── RESUMEN_PIPELINE.md
│   └── teoria/
├── models/
│   └── weapon_detection/best_model.pth
├── requirements/
├── src/
│   ├── person_extraction/
│   └── weapon_detection/
│       ├── inference/detector_pipeline.py
│       └── training/
│           ├── split_dataset.py
│           ├── augment_dataset.py
│           ├── train_fasterrcnn_light.py
│           ├── test_light_model.py
│           └── pipeline.py
├── docker-compose.yml
├── start_docker_weapons.sh / start_docker_weapons.bat
└── archive/legacy/             # Código previo, sin soporte
```

## Uso

### Entrenamiento del detector

```bash
cd src/weapon_detection/training
python pipeline.py \
    --dataset-images dataset/images \
    --dataset-xmls dataset/xmls \
    --output-dir results_light \
    --epochs 50 --batch-size 8 --amp
```

El pipeline ejecuta `split_dataset.py` → `augment_dataset.py` → `train_fasterrcnn_light.py` → `test_light_model.py`. Usa `--skip-stages` para saltar pasos ya completados.

### Pipeline de inferencia CLI

```bash
python src/weapon_detection/inference/detector_pipeline.py \
    --input data/video.mp4 \
    --output outputs/video_detected.mp4 \
    --confidence 0.55 --frame-skip 2
```

Por defecto busca `models/weapon_detection/best_model.pth` y `src/person_extraction/yolov8n.pt`. Puedes sobreescribir rutas con argumentos CLI o al instanciar `WeaponDetectionPipeline`.

### Aplicaciones web

```bash
# Image Lab (puerto 5000)
FLASK_APP=apps/image_lab/app.py flask run --reload --port 5000

# Weapon Monitor (puerto 5001)
FLASK_APP=apps/weapon_monitor/app.py flask run --reload --port 5001
```

#### API REST (Weapon Monitor)

```bash
curl -X POST "http://localhost:5001/api/detect-file" \
     -F "file=@imagen.jpg" \
     -F "confidence=0.5"
```

Endpoints adicionales: `POST /api/webcam/start`, `POST /api/webcam/stop`, `GET /api/webcam/stream` y `GET /api/health`.

## Modelos y datos

- Coloca el modelo entrenado en `models/weapon_detection/best_model.pth` (se monta automáticamente en Docker).
- El modelo YOLOv8 para personas (`yolov8n.pt`) vive en `src/person_extraction/`.
- Los datasets siguen formato Pascal VOC (`dataset/images`, `dataset/xmls`).

## Docker

El script `start_docker_weapons.sh`/`.bat` verifica los modelos, descarga YOLOv8 si falta y expone un menú para construir/levantar los contenedores.

```bash
./start_docker_weapons.sh        # Linux / macOS
start_docker_weapons.bat         # Windows
```

Manual:

```bash
docker compose build
docker compose up -d
docker compose logs -f
docker compose down
```

Servicios expuestos:
- `http://localhost:5000` → Image Lab
- `http://localhost:5001` → Weapon Monitor + API REST

## Problemas comunes

- **CUDA OOM**: baja `--batch-size`, reduce resolución o ejecuta en CPU.
- **Modelo no encontrado**: asegúrate de entrenar o copiar `best_model.pth` dentro de `models/weapon_detection/`.
- **YOLO faltante**: ejecuta los scripts de inicio o descarga manualmente `yolov8n.pt` a `src/person_extraction/`.
- **Webcam en Docker**: usa la ejecución local o comparte el dispositivo GPU/Video explícitamente.

## Limpieza y legado

Todo el código previo (`weapon_detection_pipeline/`, `weapons_detector2/`, etc.) permanece en `archive/legacy/` como referencia histórica. No se mantiene activo, pero sirve para comparación o recuperación de experimentos viejos.

## Próximos pasos

- Incorporar métricas adicionales en `apps/weapon_monitor`.
- Publicar ejemplos de consumo de la API desde otros lenguajes.
- Añadir tests automatizados para los módulos de datos y la CLI.

## Licencia y Créditos

Trabajo académico de la carrera de Procesamiento de Imágenes (UNLu).

Agradecimientos a Ultralytics (YOLOv8), PyTorch, COCO y la comunidad open-source. Se aceptan issues y PRs.
