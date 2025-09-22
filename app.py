from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from collections import Counter
import tempfile
import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO
import uuid

# Configurar matplotlib para no usar GUI
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'tu-clave-secreta-aqui'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_metadata(image_path):
    """
    Extrae metadatos básicos de la imagen.
    
    Nota: La resolución radiométrica (profundidad de bits por píxel),
    resolución espectral (número de bandas espectrales) y otros metadatos
    específicos no están disponibles en formatos de imagen comunes como
    JPG o PNG, ya que estos formatos están optimizados para visualización
    y no conservan información técnica detallada de sensores remotos.
    """
    with Image.open(image_path) as img:
        # Obtener tamaño del archivo en disco
        file_size = os.path.getsize(image_path)
        
        # Calcular resolución radiométrica (profundidad de bits)
        mode_to_bits = {
            '1': 1,      # Binario
            'L': 8,      # Escala de grises
            'P': 8,      # Paleta
            'RGB': 24,   # RGB (8 bits por canal)
            'RGBA': 32,  # RGB con alfa
            'CMYK': 32,  # CMYK
            'YCbCr': 24, # YUV
            'LAB': 24,   # LAB color space
            'HSV': 24    # HSV color space
        }
        
        bits_per_pixel = mode_to_bits.get(img.mode, 8)
        
        # Calcular rango dinámico
        max_value = (2 ** bits_per_pixel) - 1 if img.mode != 'F' else 1.0
        
        # Obtener DPI si está disponible
        dpi = img.info.get('dpi', (72, 72))  # DPI por defecto si no está especificado
        
        metadata = {
            'formato': img.format,
            'ancho': img.size[0],
            'alto': img.size[1],
            'tamaño_disco': file_size,
            'tamaño_disco_mb': round(file_size / (1024 * 1024), 2),
            'modo': img.mode,
            'canales': len(img.getbands()) if hasattr(img, 'getbands') else 1,
            
            # Resoluciones y metadatos adicionales
            'resolucion_radiometrica': bits_per_pixel,
            'resolucion_espacial_dpi': dpi,
            'resolucion_espacial_info': f"{dpi[0]} x {dpi[1]} DPI" if dpi != (72, 72) else "No especificada (72x72 DPI por defecto)",
            'resolucion_espectral': len(img.getbands()) if hasattr(img, 'getbands') else 1,
            'resolucion_temporal': "No aplica (imagen estática)",
            'rango_dinamico': f"0 - {max_value}",
            'profundidad_bits': f"{bits_per_pixel} bits por píxel",
            'tamaño_digital': f"{img.size[0]} × {img.size[1]} píxeles"
        }
        
        return metadata

def generate_histogram_image(histogram, color, channel_name):
    """
    Genera una imagen del histograma usando matplotlib.
    
    Args:
        histogram: Array con los valores del histograma
        color: Color para el gráfico (r, g, b)
        channel_name: Nombre del canal para el título
        
    Returns:
        str: Imagen en formato base64 para mostrar en HTML
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
    
    # Crear el histograma
    x_values = range(256)
    ax.fill_between(x_values, histogram, color=color, alpha=0.7, linewidth=0)
    ax.plot(x_values, histogram, color=color, linewidth=1.5)
    
    # Configurar el gráfico
    ax.set_xlim(0, 255)
    ax.set_ylim(0, max(histogram) * 1.1 if max(histogram) > 0 else 100)
    ax.set_xlabel('Valor de Píxel (0-255)', fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.set_title(f'Histograma - Canal {channel_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Configurar estilo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    
    # Limpiar la figura
    plt.close(fig)
    
    return image_base64

def analyze_channel_statistics(channel_array, color, channel_name):
    """
    Calcula estadísticas para un canal específico de la imagen y genera histograma.
    
    Args:
        channel_array: Array numpy 2D con los valores del canal
        color: Color para el histograma (formato RGB como tuple)
        channel_name: Nombre del canal para el título
        
    Returns:
        dict: Diccionario con las estadísticas del canal e imagen del histograma
    """
    # Aplanar el array para calcular estadísticas
    flat_channel = channel_array.flatten()
    
    # Calcular histograma
    histogram, _ = np.histogram(flat_channel, bins=256, range=(0, 256))
    
    # Calcular estadísticas básicas
    min_val = int(np.min(flat_channel))
    max_val = int(np.max(flat_channel))
    mean_val = float(np.mean(flat_channel))
    std_val = float(np.std(flat_channel))
    
    # Calcular moda (valor más frecuente)
    counter = Counter(flat_channel)
    mode_val = int(counter.most_common(1)[0][0])
    
    # Generar imagen del histograma
    histogram_image = generate_histogram_image(histogram, color, channel_name)
    
    return {
        'histograma': histogram.tolist(),
        'histograma_imagen': histogram_image,
        'minimo': min_val,
        'maximo': max_val,
        'promedio': round(mean_val, 2),
        'desviacion_estandar': round(std_val, 2),
        'moda': mode_val
    }

def process_image(image_path):
    """
    Procesa la imagen y extrae metadatos y estadísticas por canal.
    
    Args:
        image_path: Ruta al archivo de imagenes
        
    Returns:
        dict: Diccionario con metadatos y estadísticas por canal
    """
    # Obtener metadatos
    metadata = get_image_metadata(image_path)
    
    # Cargar imagen como array numpy
    with Image.open(image_path) as img:
        # Convertir a RGB si no está en ese formato
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convertir a array numpy
        img_array = np.array(img)
        
        # Separar canales RGB
        canal_rojo = img_array[:, :, 0]
        canal_verde = img_array[:, :, 1] 
        canal_azul = img_array[:, :, 2]
        
        # Analizar estadísticas por canal con colores específicos
        estadisticas_rojo = analyze_channel_statistics(canal_rojo, (0.8, 0.2, 0.2), "Rojo")
        estadisticas_verde = analyze_channel_statistics(canal_verde, (0.2, 0.8, 0.2), "Verde")
        estadisticas_azul = analyze_channel_statistics(canal_azul, (0.2, 0.2, 0.8), "Azul")
        
        result = {
            'metadatos': metadata,
            'canales': {
                'rojo': estadisticas_rojo,
                'verde': estadisticas_verde,
                'azul': estadisticas_azul
            }
        }
        
        return result

@app.route('/')
def index():
    """Ruta principal que muestra el formulario de carga"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Ruta POST para procesar imágenes cargadas.
    El procesamiento es síncrono y los resultados se muestran inmediatamente.
    """
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Usar archivo temporal para procesamiento
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                
                # Procesar imagen de forma síncrona
                resultados = process_image(temp_file.name)
                
                # Limpiar archivo temporal
                os.unlink(temp_file.name)
                
                # Renderizar template con resultados
                return render_template('index.html', 
                                     resultados=resultados,
                                     archivo_procesado=file.filename)
                
        except Exception as e:
            flash(f'Error al procesar la imagen: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Tipo de archivo no permitido. Use: PNG, JPG, JPEG, GIF, BMP, TIFF')
        return redirect(url_for('index'))

# TODO: Implementar procesamiento puntual de imágenes
# - Operaciones aritméticas entre imágenes
# - Transformaciones de intensidad (logarítmicas, exponenciales, etc.)

# TODO: Implementar mejora de brillo y contraste
# - Ajuste lineal de brillo/contraste
# - Ecualización de histograma
# - Estiramiento de contraste

# TODO: Implementar filtros de ruido
# - Filtro de media
# - Filtro de mediana
# - Filtros gaussianos
# - Filtros de Wiener

# TODO: Implementar detección de bordes
# - Operadores de gradiente (Sobel, Prewitt)
# - Operador Laplaciano
# - Detector de bordes Canny

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)