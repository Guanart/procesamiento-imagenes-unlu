## Plan: Sistema de Monitoreo de Armas con Alarmas y Base de Datos

Este plan fusiona la lógica de detección robusta con una arquitectura web persistente, diseñada para correr en tu hardware (Ryzen 8840HS) sobre Windows.

### Estrategia Técnica
*   **Motor de IA**: Usaremos el `WeaponDetectionPipeline` existente. Dado que la Radeon 780M no usa CUDA (NVIDIA), el sistema correrá en **CPU** por defecto. Para mantener el rendimiento en tiempo real, implementaremos *frame skipping* (procesar 1 de cada N frames).
*   **Persistencia**: SQLite para guardar configuraciones de cámaras e historial de alarmas.
*   **Concurrencia**: `threading` de Python para manejar múltiples cámaras sin bloquear el servidor web.

### Pasos de Implementación

#### 1. Capa de Datos (`apps/weapon_monitor/database.py`)
Crear un módulo para gestionar la persistencia:
*   **Tabla `cameras`**: Configuración por cámara.
    *   `id`, `name`, `source` (0, 1, url), `confidence_threshold` (ej. 0.85), `cooldown` (segundos), `consecutive_frames` (para validar detección).
*   **Tabla `alarms`**: Registro de eventos.
    *   `id`, `camera_id`, `timestamp`, `weapon_type`, `confidence`, `image_path` (full), `crop_path` (recorte).

#### 2. Motor de Monitoreo (`apps/weapon_monitor/camera_manager.py`)
Crear el gestor de cámaras que corre en segundo plano:
*   **Clase `CameraThread`**:
    *   Captura video continuamente.
    *   Ejecuta inferencia cada N frames (ajustable según carga de CPU).
    *   **Lógica Anti-Falsos Positivos**:
        1.  **Umbral**: Ignora detecciones con confianza < `confidence_threshold`.
        2.  **Consistencia**: Solo activa alarma si detecta el arma en `consecutive_frames` seguidos.
        3.  **Cooldown**: Si suena una alarma, espera `cooldown` segundos antes de permitir otra.
    *   **Acción de Alarma**: Guarda la imagen en `static/alarms/`, recorta el arma y guarda el registro en DB.

#### 3. Backend Flask (`apps/weapon_monitor/app.py`)
Actualizar la aplicación para integrar el monitor:
*   Inicializar la DB y el `CameraManager` al arranque.
*   **Nuevos Endpoints**:
    *   `GET /video_feed/<cam_id>`: Streaming MJPEG con bounding boxes dibujados.
    *   `API REST` para el frontend:
        *   `GET/POST /api/cameras`: Gestionar configuración.
        *   `GET /api/alarms`: Polling de nuevas alarmas para el popup.

#### 4. Frontend de Monitoreo (`templates/weapon_monitoring.html`)
Crear una nueva interfaz "Centro de Comando":
*   **Grilla de Cámaras**: Visualización en vivo de las cámaras activas.
*   **Panel de Configuración**: Modal para agregar cámaras y ajustar sensibilidad (Umbral, Frames, Cooldown) en caliente.
*   **Sistema de Alertas**:
    *   **Visual**: Popup/Toast rojo cuando llega una nueva alarma (vía polling a la API).
    *   **Sonoro**: Reproducir un sonido de alerta.
    *   **Historial**: Lista lateral con las últimas detecciones (clic para ver imagen guardada).

### Consideraciones para tu Hardware
*   Tu Ryzen es potente, pero la inferencia de modelos de visión es pesada.
*   **Recomendación**: Empezar con `frame_skip=3` (analizar 1 de cada 3 frames) o `frame_skip=5` si conectas múltiples cámaras, para asegurar fluidez en la UI.
