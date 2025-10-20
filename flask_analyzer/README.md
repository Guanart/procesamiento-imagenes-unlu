# Flask Analyzer

Aplicación web para análisis de imágenes RGB.

## Descripción

Servidor Flask que analiza imágenes y genera:
- Histogramas por canal RGB
- Estadísticas (min, max, promedio, moda, desviación)
- Metadatos (formato, dimensiones, tamaño)

## Uso

### Local
```bash
python app.py
```
Acceder a: `http://localhost:5000`

### Docker
```bash
docker-compose up --build
```
Acceder a: `http://localhost:5000`

## Estructura

```
flask_analyzer/
├── app.py                 # Servidor Flask
├── templates/             # Interfaz HTML
├── uploads/              # Imágenes temporales
├── Dockerfile            # Configuración Docker
└── docker-compose.yml    # Orquestación
```

## Características

- Análisis RGB por canal
- Generación de histogramas
- Interfaz con drag & drop (Tailwind CSS)
- Estadísticas detalladas por canal
