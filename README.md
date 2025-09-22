# Procesamiento de Imágenes - Universidad Nacional de Luján

Aplicación web desarrollada en Python con Flask para el análisis y procesamiento digital de imágenes. Esta aplicación permite cargar imágenes y obtener estadísticas detalladas por canal de color RGB, así como metadatos básicos.

## 🗂️ Conjunto de Datos

Este proyecto está diseñado para trabajar con el dataset de detección de armas disponible en:
**https://dasci.es/opendata/deteccion-de-armas-open-data/**

El dataset contiene imágenes de armas de diversas categorías y es utilizado para tareas de detección de objetos en el ámbito de la seguridad. Las imágenes incluyen diferentes tipos de armas en diversos contextos y condiciones de iluminación.

## 🤖 Modelo

Para futuras implementaciones de detección automática, se utilizará un modelo basado en la arquitectura **YOLO (You Only Look Once)**. 

YOLO es una red neuronal convolucional diseñada para la detección de objetos en tiempo real. Su principal característica es que procesa toda la imagen en una sola pasada (de ahí su nombre "You Only Look Once"), lo que la hace extremadamente rápida comparada con otros métodos tradicionales que requieren múltiples pasadas o ventanas deslizantes.

### Características de YOLO:
- **Velocidad**: Capaz de procesar imágenes en tiempo real
- **Precisión**: Excelente balance entre velocidad y exactitud en la detección
- **Arquitectura unificada**: Un solo modelo neuronal maneja tanto la localización como la clasificación
- **Versatilidad**: Puede detectar múltiples objetos en una sola imagen

## 📋 Funcionalidades Actuales

La aplicación actualmente implementa:

### Análisis de Metadatos
- Formato de imagen (JPG, PNG, etc.)
- Dimensiones (ancho x alto en píxeles)
- Tamaño en disco (bytes y MB)
- Modo de color y número de canales

### Análisis por Canal RGB
Para cada canal de color (Rojo, Verde, Azul):
- 📊 Histograma de distribución de píxeles
- 📉 Valor mínimo
- 📈 Valor máximo
- ➗ Promedio (media aritmética)
- 📊 Desviación estándar
- 🎯 Moda (valor más frecuente)

### Interfaz de Usuario
- Diseño moderno y responsive con Tailwind CSS
- Carga de imágenes mediante drag & drop o selección
- Visualización organizada de resultados por canal
- Tabla comparativa de estadísticas

## 🚀 Uso

### Requisitos Previos
- Docker y Docker Compose instalados en el sistema

### Ejecutar la Aplicación

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   cd procesamiento-imagenes-unlu
   ```

2. **Construir y ejecutar con Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Acceder a la aplicación:**
   Abrir el navegador y navegar a: `http://localhost:5000`

### Uso Manual (sin Docker)

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación:**
   ```bash
   python app.py
   ```

## 🛠️ Desarrollo Futuro

La aplicación incluye comentarios TODO para las siguientes funcionalidades que serán implementadas:

### Procesamiento Puntual
- Operaciones aritméticas entre imágenes
- Transformaciones de intensidad (logarítmicas, exponenciales)

### Mejora de Brillo y Contraste
- Ajuste lineal de brillo/contraste
- Ecualización de histograma
- Estiramiento de contraste

### Filtros de Ruido
- Filtro de media
- Filtro de mediana
- Filtros gaussianos
- Filtros de Wiener

### Detección de Bordes
- Operadores de gradiente (Sobel, Prewitt)
- Operador Laplaciano
- Detector de bordes Canny

## 🏗️ Estructura del Proyecto

```
procesamiento-imagenes-unlu/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias de Python
├── Dockerfile            # Configuración del contenedor
├── docker-compose.yml    # Orquestación de servicios
├── templates/
│   └── index.html        # Plantilla HTML con Tailwind CSS
├── uploads/              # Directorio para archivos temporales
└── README.md            # Este archivo
```

## 🔧 Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Procesamiento**: Pillow, NumPy
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Containerización**: Docker, Docker Compose

## 📝 Notas Técnicas

> **Importante**: La resolución radiométrica (profundidad de bits por píxel), resolución espectral (número de bandas espectrales) y otros metadatos específicos no están disponibles en formatos de imagen comunes como JPG o PNG, ya que estos formatos están optimizados para visualización y no conservan información técnica detallada de sensores remotos.

## 👥 Autor

Proyecto desarrollado para la asignatura de Procesamiento Digital de Imágenes - Universidad Nacional de Luján.

---
*Desarrollado con ❤️ usando Flask, Python, Pillow y NumPy*
