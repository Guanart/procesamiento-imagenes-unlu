# Procesamiento de Imágenes - UNLu

Proyecto de procesamiento de imágenes con detección de personas y armas.

## Estructura del Proyecto

```
procesamiento-imagenes-unlu/
├── weapons_augmenter/      # Aumentación de dataset de armas
├── person_extraction/      # Extracción y mejora de personas
├── flask_analyzer/         # Análisis RGB de imágenes
└── docs/                   # Documentación técnica
```

## Módulos

### 1. weapons_augmenter
Aumenta el dataset de armas (pistolas y cuchillos) aplicando transformaciones simples.

**Uso:**
```bash
cd weapons_augmenter
python simple_augmenter.py
python normalize.py --input dataset/augmented --output dataset/normalized --size 224 224
```

**Resultado:** 1,420 imágenes → 5,680 imágenes (4x)

### 2. person_extraction
Pipeline de extracción y mejora de personas desde videos usando YOLOv8.

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
cd weapons_classifier
python train_weapons_classifier.py --data_dir dataset/normalized --output_dir results --epochs 10 --batch_size 32
python test_weapons_classifier.py --image /ruta/a/imagen.jpg --model results/best_model.pth
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Dependencias Principales

- OpenCV 4.12
- YOLOv8 (Ultralytics)
- Flask 3.1
- NumPy, SciPy

## Documentación

Ver carpeta `docs/` para documentación técnica detallada.
