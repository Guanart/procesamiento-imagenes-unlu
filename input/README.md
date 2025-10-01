# Directorio de Videos de Entrada

Este directorio contiene los videos que serán procesados por el sistema de detección de personas.

## Instrucciones de Uso

1. **Coloca tu video aquí** con el nombre `test_video.mp4`, o
2. **Especifica la ruta** al ejecutar el script:
   ```bash
   python video_processor.py --video input/mi_video.mp4
   ```

## Formatos Soportados

El sistema soporta los siguientes formatos de video:
- **MP4** (.mp4) - Recomendado
- **AVI** (.avi)
- **MOV** (.mov)
- **MKV** (.mkv)
- **WMV** (.wmv)

## Recomendaciones de Calidad

Para obtener mejores resultados en la detección:

### Resolución
- **Mínima**: 480p (640x480)
- **Recomendada**: HD (1280x720) o superior
- **Óptima**: Full HD (1920x1080)

### Frame Rate
- **Mínimo**: 15 FPS
- **Recomendado**: 30 FPS
- **Máximo procesable**: Sin límite (el sistema se adapta)

### Condiciones de Grabación
- ✅ **Buena iluminación**: Evita videos muy oscuros
- ✅ **Cámara estable**: Reduce blur y falsos positivos
- ✅ **Personas visibles**: Al menos 50% del cuerpo visible
- ✅ **Distancia adecuada**: Personas de al menos 30px de alto

### Consideraciones del Dataset
- 📹 **Videos con personas armadas**: ✅ Soportado
- 📹 **Videos sin armas**: ✅ Soportado  
- 📹 **Múltiples personas**: ✅ Detecta todas las personas
- 📹 **Escenas complejas**: ✅ Funciona en entornos diversos

## Ejemplos de Videos Ideales

### Escenarios de Seguridad
- Cámaras de vigilancia en edificios
- Videos de cámaras corporales policiales
- Grabaciones de sistemas de seguridad
- Videos de entrenamiento de personal

### Escenarios Públicos
- Videos de calles y plazas
- Grabaciones en transporte público
- Eventos deportivos o culturales
- Videos de cámaras de tráfico

## Privacidad y Ética

⚠️ **Importante**: Asegúrate de cumplir con todas las regulaciones de privacidad aplicables:
- Consentimiento para grabación de personas
- Cumplimiento con GDPR/LOPD según jurisdicción
- Uso apropiado de datos biométricos
- Anonización de resultados cuando sea necesario

## Limitaciones Técnicas

### No Recomendado:
- ❌ Videos con resolución menor a 320x240
- ❌ Videos extremadamente borrosos
- ❌ Grabaciones nocturnas sin iluminación adecuada
- ❌ Videos con compresión muy alta (calidad muy baja)

### Casos Especiales:
- 🟡 **Videos largos**: Pueden tomar considerable tiempo de procesamiento
- 🟡 **Videos 4K**: Requieren más memoria RAM
- 🟡 **Alta densidad de personas**: Puede reducir velocidad de procesamiento

---

**Tip**: Si no tienes un video de prueba, puedes descargar videos de ejemplo desde repositorios públicos como:
- [Sample Videos](https://sample-videos.com/)
- [Pixabay Videos](https://pixabay.com/videos/)
- [Pexels Videos](https://www.pexels.com/videos/)

Recuerda verificar las licencias de uso antes de utilizar videos de terceros.