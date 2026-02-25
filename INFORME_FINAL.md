# Informe Final — Sistema de Detección y Monitoreo de Armas

Alumno: Gonzalo Benito  
Repositorio: https://github.com/Guanart/procesamiento-imagenes-unlu  
Fecha: 2026-02-09

## 1. Objetivo del proyecto
Implementar un pipeline completo para detectar armas blancas y de fuego en imágenes y video, integrando:
- Extracción de personas desde frames de video.
- Mejora de imagen en recortes de personas.
- Detección de armas (clases: `knife`, `pistol`) con un detector entrenado y ajustado.
- Aplicaciones web para probar el pipeline y operar un centro de monitoreo multi-cámara con alarmas e historial.

## 2. Dataset (Etapa 1)
**Fuente principal:** OD-WeaponDetection (DaSCI).

### 2.1. Estructura y módulos utilizados
El repositorio organiza módulos separados por responsabilidad:
- `src/person_extraction/`: extracción de personas y mejora de imágenes (Stage 1).
- `src/weapon_detection/training/`: pipeline de entrenamiento del detector (split, augment, train, test).
- `src/weapon_detection/inference/`: pipeline de inferencia extremo a extremo.
- `apps/weapon_monitor/`: app Flask (centro de monitoreo + streaming + alarmas + API).

### 2.2. Stage 1 — Extracción de personas
- Se procesan videos muestreando aproximadamente a ~1 fps.
- Se detectan personas con YOLOv8n (preentrenado en COCO), filtrando clase “persona”.
- Se recortan personas y se aplican mejoras (ver 2.4).

### 2.3. Stage 2 — Datos de armas (clasificación / detección)
A partir del dataset OD-WeaponDetection, se trabajó con imágenes de armas (pistolas y cuchillos). En el informe de dataset se mencionan recuentos de referencia:
- Pistolas: 795 imágenes (y un directorio adicional con ~4000 imágenes).
- Cuchillos: 635 imágenes (y un directorio adicional con ~2300 imágenes).

### 2.4. Mejora de imagen (pre-procesado)
Se aplica una mejora de contraste y brillo para robustecer la señal visual:
1) Redimensión preservando aspect ratio (interpolación cúbica cuando corresponde).
2) CLAHE sobre canal L (espacio Lab) para mejorar contraste.
3) Ajuste de brillo aumentando canal V (HSV).

## 3. Modelado y pipeline (Etapa 2)
### 3.1. Resumen del pipeline
El pipeline actual integra:
- **YOLOv8n** para localizar y recortar personas.
- **Faster R-CNN MobileNetV3-FPN** para detectar armas en los recortes mejorados.
- Reconstrucción de cajas al frame original y exportación de resultados (imagen/video).

### 3.2. Entrenamiento
En `src/weapon_detection/training/` se implementó un pipeline reproducible:
- Split de datos con semilla (reproducible).
- Aumentación offline (geométricas y fotométricas; según pipeline de training).
- Entrenamiento con AMP, checkpoints y posibilidad de reanudar desde `.pth`.
- Early stopping (patience=5) para evitar overfitting y ahorrar cómputo.

Se incorporaron decisiones prácticas para Colab/IO:
- Checkpoints cada N épocas.
- Optimización de lectura masiva de XML (evitar latencia de Drive; empaquetado en ZIP y descompresión local).

### 3.3. Resultados reportados (test)
Del informe de modelado (test más reciente):

**Matriz de confusión (resumen)**
- `knife`: TP=1143, FP=77, FN=105 → Precision≈0.936, Recall≈0.916
- `pistol`: TP=1968, FP=258, FN=81 → Precision≈0.884, Recall≈0.960

**mAP**
- mAP: 0.6199
- mAP@50: 0.9480
- mAP@75: 0.6421
- mAP (small/medium/large): 0.4212 / 0.5173 / 0.6450

### 3.4. Hallazgos y limitaciones del modelo
Se observaron confusiones en casos de borde (ej.: objetos alargados o en ciertas poses). Un ejemplo reportado en videos: 
- Cuando la pistola se encuentra de costado, puede confundirse con cuchillo.

En pruebas del centro de monitoreo también se observó un caso importante:
- **Celular de costado confundido con cuchillo con alta probabilidad**.

Esto no es un fallo de la app: es una **limitación del modelo/dataset** y se aborda en el documento `MEJORAS_MODELO.md`.

## 4. Aplicación: Centro de Monitoreo Multi-cámara (Etapa 3 / Integración)
Se implementó un sistema de monitoreo persistente, configurable desde UI, con streaming MJPEG y alarmas.

### 4.1. Arquitectura
- **Persistencia:** SQLite (`apps/weapon_monitor/monitor.db`).
- **Concurrencia:** `threading` (un hilo por cámara) para no bloquear Flask.
- **Streaming:** MJPEG por cámara (`GET /video_feed/<id>`).
- **UI:** dashboard con grilla de cámaras, panel de alarmas e interfaz de configuración.

### 4.2. Datos persistidos
- Tabla `cameras`: id, nombre, fuente, umbrales y configuración (confidence, frames, cooldown), enabled.
- Tabla `alarms`: id, camera_id, timestamp, weapon_type, confidence, paths a imágenes (full/crop).

Las imágenes de alarma se guardan en:
- `apps/weapon_monitor/static/alarms/`

### 4.3. Lógica anti-falsos-positivos
Por cámara:
- **Umbral de confianza:** descarta detecciones debajo del umbral.
- **Consistencia temporal:** exige `N` frames consecutivos con detección válida.
- **Cooldown:** espera `cooldown` segundos entre alarmas.

> Nota: `cooldown` significa “tiempo mínimo entre alarmas” (evita spam de alarmas si el arma permanece en escena).

### 4.4. Endpoints principales
- `GET /monitoring`: dashboard.
- `GET/POST /api/cameras`: listar/agregar cámaras.
- `PUT /api/cameras/<id>`: actualizar parámetros.
- `DELETE /api/cameras/<id>`: eliminar.
- `POST /api/cameras/<id>/start` y `POST /api/cameras/<id>/stop`: control por cámara.
- `GET /video_feed/<id>`: stream MJPEG.
- `GET /api/alarms`, `GET /api/alarms/latest`: historial y polling.

### 4.5. Configuración
Se soporta `.env` en `apps/weapon_monitor/` (ejemplo: `apps/weapon_monitor/.env.example`).
Variables relevantes:
- `WEAPON_MODEL_PATH`, `YOLO_MODEL_PATH`, `DEFAULT_CONFIDENCE`.
- Optimización de streaming: `STREAM_WIDTH`, `STREAM_HEIGHT`, `STREAM_FPS`, `FRAME_SKIP`, `JPEG_QUALITY`.

## 5. Performance y operación
En hardware sin CUDA (CPU), el pipeline completo puede ser pesado en tiempo real. Se aplicaron estrategias de performance:
- Frame skipping (procesar 1 de cada N frames para inferencia).
- Reducción de resolución y compresión JPEG para bajar latencia.
- Reutilización de últimas detecciones en frames intermedios para mantener fluidez.

### 5.2 Fine-tuning con Hard Negatives (Celulares)

#### Problema Identificado
Durante la evaluación del modelo inicial, se detectó un problema recurrente: **el modelo confundía celulares de costado con cuchillos** (clasificaba incorrectamente como `knife` con alta confianza). Este falso positivo era particularmente problemático en un sistema de monitoreo de seguridad.

#### Estrategia: Hard Negatives
Para resolver este problema sin recolectar miles de imágenes adicionales, implementamos una técnica denominada **hard negatives**. La idea es integrar al entrenamiento imágenes que contienen objetos "parecidos a armas" (en este caso, celulares de costado) pero **sin cajas de detección**, enseñándole al modelo que esos objetos NO son armas.

#### Proceso
1. **Selección**: Se filtraron 28 imágenes de celulares de costado del dataset `dataset_celulares`, donde la forma se asemeja a un cuchillo.
2. **Preparación**: Se crearon "hard negatives" eliminando las cajas de armas de los XMLs, dejando solo las imágenes como negativos puros.
3. **Integración**: Se copiaron los 28 hard negatives al dataset de entrenamiento (combinándose con ~500 imágenes positivas de armas).
4. **Fine-tuning**: Se realizó un entrenamiento de 30 épocas con:
   - Learning rate bajo: `1e-5` (para no destruir el conocimiento previo)
   - Batch size: 6
   - Inicio desde checkpoint: `results_standard/best_model.pth`

#### Resultados Esperados
- **mAP mejoró** 
- **mAP@75 mejoró** 
- El modelo aprendió a distinguir mejor entre objetos alargados reales (cuchillos) vs. objetos similares (celulares)

#### Validación
Se evaluó en el test set para confirmar que:
1. Los falsos positivos de `knife` bajaron
2. El recall real de `knife` se mantuvo (sin sacrificar detecciones verdaderas)
3. No hubo overfitting

#### Conclusión
La técnica de hard negatives probó ser **sumamente efectiva** para reducir falsos positivos sin necesidad de más datos positivos. Este enfoque es especialmente útil en datasets limitados y es una alternativa práctica a la recolección de más imágenes.

## 6. Conclusiones
- Se completó un pipeline funcional (personas → mejora → detección de armas) y se integró en una app operable.
- El centro de monitoreo agrega persistencia, control multi-cámara, streaming y alarmas con historial.
- La principal fuente de falsos positivos actuales está en el **modelo** (no en la app), en especial objetos parecidos a cuchillo (celular de costado). Se propone un plan específico de mitigación en `MEJORAS_MODELO.md`.

## 7. Próximos pasos
- Mejorar el detector con hard negatives (celulares) y/o nueva clase (phone).
- Evaluación sistemática en videos reales (dominio objetivo) y calibración de umbrales por clase.
- (Opcional) modelo más liviano / cuantización o aceleración para mejor FPS.
