# Procesamiento de Imágenes - UNLU

Aplicación web para análisis de imágenes digitales desarrollada para la materia "Procesamiento de Imágenes" de la Universidad Nacional de Luján.

## Descripción

Esta aplicación Flask permite subir imágenes y obtener un análisis completo que incluye:
- **Extracción de metadatos EXIF** completos
- **Análisis estadístico por canal RGB**: mínimo, máximo, media, desviación estándar, mediana y moda  
- **Generación de histogramas** interactivos por canal
- **Interface web moderna** con Tailwind CSS
- **API REST** para integración programática

## Funcionalidades Principales

### 📊 Análisis Estadístico RGB
- Cálculo de estadísticas descriptivas por canal (R, G, B)
- Generación automática de histogramas de distribución
- Identificación de valores modales y frecuencias
- Visualización gráfica de la información

### 📷 Extracción de Metadatos
- Lectura completa de datos EXIF
- Información de cámara y configuración de captura
- Timestamps y datos de geolocalización (si disponibles)
- Detalles técnicos del archivo de imagen

### 🎨 Interface de Usuario
- Diseño responsive con Tailwind CSS
- Drag & drop para subida de archivos
- Visualización clara y organizada de resultados
- Soporte para múltiples formatos de imagen

## Requisitos del Sistema

- Python 3.11+
- Docker y Docker Compose
- 2GB RAM mínimo recomendado
- Espacio libre para procesamiento de imágenes

## Instalación

### Opción 1: Docker (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/Guanart/procesamiento-imagenes-unlu.git
cd procesamiento-imagenes-unlu

# Construir y ejecutar con Docker Compose
docker-compose up --build

# La aplicación estará disponible en http://localhost:5000
```

### Opción 2: Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/Guanart/procesamiento-imagenes-unlu.git
cd procesamiento-imagenes-unlu

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

## Uso

### Interface Web
1. Acceder a `http://localhost:5000`
2. Subir una imagen mediante drag & drop o selección de archivo
3. Revisar los resultados del análisis en la página de resultados

### API REST

```bash
# Análisis de imagen via API
curl -X POST -F "file=@imagen.jpg" http://localhost:5000/api/analyze
```

Respuesta JSON con metadatos y análisis RGB completo.

## Formatos Soportados

- **PNG** - Portable Network Graphics
- **JPEG/JPG** - Joint Photographic Experts Group  
- **GIF** - Graphics Interchange Format
- **BMP** - Windows Bitmap
- **TIFF** - Tagged Image File Format

Tamaño máximo: 16MB por archivo

## Referencias Académicas

### Datasets de Referencia
- **DaSCI-es**: Datasets científicos disponibles en [dasci.es](https://dasci.es) para investigación en procesamiento de imágenes
- Colecciones de imágenes etiquetadas para machine learning
- Datasets de imágenes médicas y científicas

### Modelos y Algoritmos
- **YOLO (You Only Look Once)**: Algoritmo de detección de objetos en tiempo real
- Técnicas de análisis estadístico en espacios de color RGB
- Métodos de extracción y análisis de metadatos EXIF
- Algoritmos de generación de histogramas y análisis de distribución

### Fundamentos Teóricos
- Procesamiento digital de imágenes (Gonzalez & Woods)
- Análisis estadístico de señales digitales
- Espacios de color y representación digital
- Técnicas de visualización de datos

## Estructura del Proyecto

```
procesamiento-imagenes-unlu/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias Python
├── Dockerfile            # Configuración Docker
├── docker-compose.yml    # Orquestación de servicios
├── templates/            # Plantillas HTML
│   ├── base.html         # Template base
│   ├── index.html        # Página principal
│   └── results.html      # Página de resultados
└── uploads/              # Directorio temporal de imágenes
```

## Tecnologías Utilizadas

- **Flask** - Framework web de Python
- **Pillow (PIL)** - Manipulación de imágenes
- **OpenCV** - Procesamiento de imágenes
- **NumPy** - Computación numérica
- **Matplotlib** - Generación de gráficos
- **SciPy** - Análisis estadístico
- **Tailwind CSS** - Framework CSS
- **Docker** - Containerización

## Mejoras Futuras (TODO)

### Funcionalidades
- [ ] Soporte para canal alpha en imágenes PNG
- [ ] Análisis en espacios de color HSV y LAB
- [ ] Detección automática de objetos con YOLO
- [ ] Análisis de texturas y patrones
- [ ] Comparación entre múltiples imágenes
- [ ] Exportación de reportes en PDF

### Técnicas
- [ ] Integración con modelos YOLO pre-entrenados
- [ ] Análisis de calidad de imagen (nitidez, ruido)
- [ ] Detección de metadatos manipulados
- [ ] Análisis forense de imágenes digitales
- [ ] Clustering automático de colores dominantes

### Arquitectura
- [ ] Procesamiento asíncrono con Celery/Redis
- [ ] Base de datos para histórico de análisis
- [ ] Autenticación y autorización de usuarios
- [ ] API GraphQL para consultas complejas
- [ ] Microservicios para análisis específicos
- [ ] Deploy con Kubernetes

### Seguridad
- [ ] Validación avanzada de tipos MIME
- [ ] Sandboxing para procesamiento de archivos
- [ ] Límites de rate limiting por usuario
- [ ] Encriptación de archivos temporales
- [ ] Logs de auditoría de accesos

## Contribuir

1. Fork del repositorio
2. Crear branch para nueva funcionalidad
3. Implementar cambios con tests
4. Enviar Pull Request

## Licencia

Este proyecto es de uso académico para la Universidad Nacional de Luján.

## Contacto

Desarrollo para la materia Procesamiento de Imágenes - UNLU
- Universidad Nacional de Luján
- Departamento de Ciencias Básicas
