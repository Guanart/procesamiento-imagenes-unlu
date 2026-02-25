# Centro de Monitoreo de Armas

Este módulo agrega un sistema de vigilancia multi-cámara con alarmas, persistencia en SQLite y configuración desde la UI.

## Cómo iniciar

1. Crear (opcional) archivo `.env` en `apps/weapon_monitor/` tomando como base `.env.example`:

```
WEAPON_MODEL_PATH=/home/gbenito/universidad/procesamiento-imagenes-unlu/models/best_model.pth
YOLO_MODEL_PATH=/home/gbenito/universidad/procesamiento-imagenes-unlu/src/person_extraction/yolov8n.pt
DEFAULT_CONFIDENCE=0.5
```

2. Ejecutar la app:

```bash
WEAPON_MODEL_PATH=/home/gbenito/universidad/procesamiento-imagenes-unlu/models/best_model.pth \
  /home/gbenito/universidad/procesamiento-imagenes-unlu/.venv/bin/python \
  /home/gbenito/universidad/procesamiento-imagenes-unlu/apps/weapon_monitor/app.py
```

Si existe un `.env` en la carpeta `apps/weapon_monitor/`, se carga automáticamente.

3. Abrir `http://localhost:5001` y entrar a **Centro de Monitoreo**.

## Cómo agregar cámaras

En el botón **Configurar Cámaras**:

- **Nombre**: Texto libre para identificar la cámara.
- **Fuente (0, 1, URL)**:
  - `0`, `1`, `2` ... → Índice de webcam local en Windows (tu cámara integrada suele ser `0`).
  - `rtsp://...` → URL RTSP de una cámara IP (ejemplo: `rtsp://usuario:clave@192.168.0.10:554/stream1`).
  - `http://...` o `https://...` → Fuentes HTTP con MJPEG/Video (según soporte de la cámara y OpenCV).
  - (Opcional) Ruta a archivo de video → `C:/videos/camara1.mp4` (útil para pruebas, no en vivo).
- **Conf (0.85 por defecto recomendado)**: Umbral mínimo de confianza para considerar una detección de arma. Subirlo ayuda a evitar confundir celulares por cuchillos.
- **Frames (5 recomendado)**: Cantidad de frames consecutivos con detección válida requeridos antes de disparar la alarma.

Al guardar, el sistema recarga y **levanta todas las cámaras habilitadas**. Las imágenes de alarma se guardan en `apps/weapon_monitor/static/alarms/`.

## Lógica Anti-Falsos Positivos

- **Umbral alto**: Detecciones con confianza menor al umbral se descartan.
- **Consistencia temporal**: Se exige que el arma se detecte durante `N` frames consecutivos.
- **Cooldown**: Tras una alarma, se espera `cooldown` segundos antes de permitir otra.

Valores iniciales (por cámara):
- `confidence_threshold = 0.85`
- `consecutive_frames = 5`
- `cooldown = 10` segundos

## API

- `GET /api/cameras` → Lista de cámaras.
- `POST /api/cameras` → Agregar cámara.
- `DELETE /api/cameras/<id>` → Eliminar cámara.
- `PUT /api/cameras/<id>` → Actualizar parámetros.
- `GET /video_feed/<id>` → Stream MJPEG de la cámara procesada.
- `GET /api/alarms` → Últimas alarmas.
- `GET /api/alarms/latest` → Timestamp de la última alarma (para polling).

## Notas de compatibilidad

- En **Windows con AMD Radeon 780M**, el sistema corre en **CPU**; CUDA no está disponible.
- Si utilizas varias cámaras en CPU, ajusta el **frame skipping** (interno a `camera_manager.py`) para mantener fluidez.

## Ubicaciones relevantes

- Base de datos: `apps/weapon_monitor/monitor.db`
- Alarmas (imágenes): `apps/weapon_monitor/static/alarms/`
- Configuración opcional: `apps/weapon_monitor/.env`
