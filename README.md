# Sistema de Detección de Armas en Tiempo Real



Sistema completo de detección de armas en imágenes y videos utilizando Deep Learning. El sistema identifica personas en el contenido multimedia, mejora las imágenes y detecta armas (cuchillos y pistolas) con modelos de detección de objetos.Proyecto de detección de armas (pistola/cuchillo) en personas usando Faster R-CNN optimizado para AMD ROCm (Radeon 780M iGPU).



## 📋 Índice---



- [Descripción General](#descripción-general)## 🚀 Inicio Rápido

- [Arquitectura del Sistema](#arquitectura-del-sistema)

- [Módulos del Proyecto](#módulos-del-proyecto)### 1. Verificar GPU AMD

- [Instalación](#instalación)```bash

- [Uso](#uso)python check_gpu.py

- [Entrenamiento del Modelo](#entrenamiento-del-modelo)```

- [Aplicación Web](#aplicación-web)

- [Docker](#docker)### 2. Entrenar detector

# 🔫 Sistema de Detección de Armas en Imágenes y Video

Proyecto académico UNLu para detección de armas (pistola/cuchillo) en personas usando Deep Learning (Faster R-CNN + YOLOv8 + Flask). Optimizado para GPU AMD/ROCm y CUDA.

---

## 📋 Índice

- [Descripción General](#descripción-general)
- [Arquitectura](#arquitectura)
- [Módulos](#módulos)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Entrenamiento](#entrenamiento)
- [API y Web](#api-y-web)
- [Docker](#docker)
- [Requisitos y Dependencias](#requisitos-y-dependencias)
- [Estructura](#estructura)
- [Problemas Comunes](#problemas-comunes)
- [Licencia y Créditos](#licencia-y-créditos)

---

## 🎯 Descripción General

El sistema procesa videos e imágenes en tres etapas:
1. **Extracción de Personas**: YOLOv8 detecta personas y recorta cada una.
2. **Mejora de Imagen**: CLAHE y ajuste de brillo sobre cada recorte.
3. **Detección de Armas**: Faster R-CNN identifica armas (knife, pistol) en los recortes.

Características:
- Detección de personas (YOLOv8n, COCO)
- Mejora automática de imágenes
- Detección de armas (Faster R-CNN, MobileNetV3)
- Aplicación web Flask (drag & drop, webcam)
- Pipeline CLI y API REST
- Despliegue con Docker
- Optimizado para GPU (AMP) y CPU

---

## 🏗️ Arquitectura

```
Video/Imagen → [YOLOv8 Person Detection] → [Image Enhancement] → [Faster R-CNN Weapon Detection] → Resultado Anotado
```

---

## 📦 Módulos

### 1. person_extraction/
Extracción de personas desde videos/imágenes usando YOLOv8.
- `video_processor.py`: Procesa videos, detecta personas, genera recortes
- `image_enhancer.py`: Mejora imágenes con CLAHE y brillo
- `yolov8n.pt`: Modelo YOLOv8 nano

### 2. weapons_detector2/
Entrenamiento y evaluación del detector de armas.
- `train_fasterrcnn_light.py`: Entrenamiento Faster R-CNN
- `test_light_model.py`: Inferencia en imágenes
- `dataset/`: Imágenes + XML (Pascal VOC)
- `results_light/best_model.pth`: Modelo entrenado

### 3. weapon_detection_pipeline/
Pipeline integrado: personas → mejora → armas.
- `pipeline.py`: Orquestador CLI y programático

### 4. flask_analyzer/
Aplicación web Flask (drag & drop, webcam, API REST).
- `weapon_detector_app.py`: Servidor Flask
- `templates/`: HTML

---

## 🔧 Instalación

### Opción A: Docker (recomendado)
```bash
./start_docker_weapons.sh   # Linux/Mac
start_docker_weapons.bat    # Windows
```
Acceder a http://localhost:5001

### Opción B: Local
```bash
git clone <repo>
cd procesamiento-imagenes-unlu
python -m venv venv
source venv/bin/activate
pip install -r requirements_weapon_detector.txt
```

---

## 🎮 Uso Rápido

```bash
# Imagen
python weapon_detection_pipeline/pipeline.py --mode image --input test.jpg --output out.jpg
# Video
python weapon_detection_pipeline/pipeline.py --mode video --input in.mp4 --output out.mp4 --frame-skip 5
# Web
python flask_analyzer/weapon_detector_app.py
```

---

## 🧠 Entrenamiento

```bash
cd weapons_detector2
python train_fasterrcnn_light.py --epochs 25 --batch-size 16 --amp
```
Hipótesis óptimas: 320x320 resize, batch 16, AMP activo, 25–50 épocas.

---

## 🌐 API y Web

**API REST:**
```bash
curl -X POST http://localhost:5001/api/detect-file -F "file=@imagen.jpg" -F "confidence=0.5"
```
Endpoints webcam: `/api/webcam/start`, `/api/webcam/stop`, `/api/webcam/stream`.

**Web:**
Abrir http://localhost:5001 y usar los módulos de archivos o webcam.

---

## 🐳 Docker

```bash
docker-compose build --no-cache
docker-compose up -d
docker-compose logs -f
docker-compose down
```
Volúmenes: modelo y resultados se actualizan sin reconstruir.

---

## 📋 Requisitos y Dependencias

**Archivo principal:** `requirements_weapon_detector.txt`

Por defecto, sólo incluye dependencias mínimas para INFERENCIA y Web (producción):

```text
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
Flask>=2.3.0
```

Extras para ENTRENAMIENTO (descomentar si se entrena el modelo):

```text
# torchmetrics>=1.0.0       # mAP y métricas avanzadas
# tqdm>=4.65.0              # Barras de progreso
# psutil>=5.9.0             # Monitoreo de recursos
# scikit-learn>=1.2.0       # Métricas adicionales / split
# matplotlib>=3.7.0         # Gráficos de entrenamiento
# scipy>=1.10.0             # Dependencias científicas
```

Esto acelera la instalación y reduce el tamaño de la imagen Docker. Si vas a entrenar, instala los extras:

```bash
pip install torchmetrics tqdm psutil scikit-learn matplotlib scipy
```

---

## 📂 Estructura

```
procesamiento-imagenes-unlu/
├── README.md
├── docker-compose.yml
├── start_docker_weapons.sh / .bat
├── requirements_weapon_detector.txt
├── person_extraction/
├── weapons_detector2/
├── weapon_detection_pipeline/
└── flask_analyzer/
```

---

## 🐛 Problemas Comunes

- CUDA out of memory → bajar batch o usar CPU
- Modelo no encontrado → entrenar y ubicar en `weapons_detector2/results_light/`
- Webcam en Docker → usar modo local

---

## 📝 Licencia y Créditos

Trabajo académico - Procesamiento de Imágenes - UNLu.

Agradecimientos: YOLOv8 (Ultralytics), PyTorch, COCO, Google Colab.

**Última actualización:** Noviembre 2025


```### Salida
