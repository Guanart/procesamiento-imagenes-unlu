# API Documentation

## Image Analysis API

### Upload and Analyze Image

**Endpoint:** `POST /api/analyze`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file`: Image file (required)
  - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
  - Maximum size: 16MB

**Example Request:**
```bash
curl -X POST \
  -F "file=@sample.jpg" \
  http://localhost:5000/api/analyze
```

**Success Response (200):**
```json
{
  "metadata": {
    "formato": "JPEG",
    "modo": "RGB", 
    "tamaño": [800, 600],
    "ancho": 800,
    "alto": 600,
    "exif": {
      "DateTime": "2024:01:15 10:30:00",
      "Make": "Canon",
      "Model": "EOS R5"
    }
  },
  "rgb_analysis": {
    "rojo": {
      "minimo": 0,
      "maximo": 255,
      "media": 127.5,
      "desviacion_estandar": 73.2,
      "mediana": 128.0,
      "moda": 130,
      "frecuencia_moda": 1250,
      "varianza": 5358.24,
      "histograma": {
        "valores": [10, 15, 20, ...],
        "bins": [0, 1, 2, 3, ...]
      }
    },
    "verde": { ... },
    "azul": { ... },
    "histograma_grafico": "base64_encoded_image_string"
  },
  "filename": "sample.jpg"
}
```

**Error Response (400):**
```json
{
  "error": "No file provided"
}
```

**Error Response (400):**
```json
{
  "error": "Invalid file type"
}
```

**Error Response (500):**
```json
{
  "error": "Error processing image: [detail]"
}
```

## Web Interface

### Home Page
- **URL:** `/`
- **Method:** GET
- **Description:** Main page with file upload form

### Upload Image
- **URL:** `/upload`
- **Method:** POST
- **Content-Type:** `multipart/form-data`
- **Description:** Upload image via web form
- **Response:** Redirects to results page

## Statistics Explanation

### RGB Channel Analysis
Each color channel (Red, Green, Blue) is analyzed separately:

- **Mínimo/Máximo**: Range of intensity values (0-255)
- **Media**: Average intensity value
- **Desviación Estándar**: Standard deviation of intensity values
- **Mediana**: Middle value when intensities are sorted
- **Moda**: Most frequent intensity value
- **Varianza**: Square of standard deviation
- **Histograma**: Distribution of intensity values across 256 bins

### Metadata
EXIF data extracted from image files includes:
- Camera information (make, model)
- Capture settings (ISO, aperture, shutter speed)
- Timestamps
- GPS coordinates (if available)
- Image processing settings

## TODO: Future API Enhancements

- [ ] Authentication with API keys
- [ ] Rate limiting per user/IP
- [ ] Batch processing endpoint
- [ ] Webhook notifications
- [ ] GraphQL query interface
- [ ] Real-time WebSocket updates
- [ ] Image comparison endpoints
- [ ] YOLO object detection integration
- [ ] Custom analysis profiles
- [ ] Export formats (PDF, CSV, XML)