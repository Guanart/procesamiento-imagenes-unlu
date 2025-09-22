"""
Flask App para Análisis de Imágenes - Procesamiento de Imágenes UNLU
Analiza metadatos y estadísticas RGB de imágenes subidas
"""
import os
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidor
import matplotlib.pyplot as plt
from scipy import stats
import exifread
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu-clave-secreta-aqui'  # TODO: Usar variable de entorno
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_metadata(image_path):
    """
    Extraer metadatos de la imagen usando EXIF
    TODO: Añadir más formatos de metadatos (IPTC, XMP)
    """
    metadata = {}
    
    try:
        # Usar PIL para metadatos básicos
        with Image.open(image_path) as img:
            metadata['formato'] = img.format
            metadata['modo'] = img.mode
            metadata['tamaño'] = img.size
            metadata['ancho'] = img.width
            metadata['alto'] = img.height
            
            # Extraer datos EXIF si existen
            exifdata = img.getexif()
            if exifdata:
                exif_info = {}
                for tag_id, value in exifdata.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_info[tag] = str(value)
                metadata['exif'] = exif_info
    
        # Usar exifread para metadatos más detallados
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if tags:
                detailed_exif = {}
                for key, value in tags.items():
                    if key not in ['JPEGThumbnail', 'TIFFThumbnail']:
                        detailed_exif[key] = str(value)
                metadata['exif_detallado'] = detailed_exif
                
    except Exception as e:
        metadata['error_metadatos'] = str(e)
    
    return metadata

def analyze_rgb_channels(image_path):
    """
    Analizar estadísticas de cada canal RGB
    Calcula: mínimo, máximo, media, desviación estándar, moda, histograma
    TODO: Añadir análisis de canal alpha si existe
    TODO: Soporte para espacios de color HSV, LAB
    """
    try:
        # Cargar imagen con OpenCV (BGR por defecto)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Separar canales
        red_channel = img_rgb[:, :, 0].flatten()
        green_channel = img_rgb[:, :, 1].flatten()
        blue_channel = img_rgb[:, :, 2].flatten()
        
        channels = {
            'rojo': red_channel,
            'verde': green_channel,
            'azul': blue_channel
        }
        
        statistics = {}
        
        for color_name, channel_data in channels.items():
            # Calcular estadísticas básicas
            channel_stats = {
                'minimo': int(np.min(channel_data)),
                'maximo': int(np.max(channel_data)),
                'media': float(np.mean(channel_data)),
                'desviacion_estandar': float(np.std(channel_data)),
                'varianza': float(np.var(channel_data)),
                'mediana': float(np.median(channel_data))
            }
            
            # Calcular moda
            try:
                mode_result = stats.mode(channel_data, keepdims=True)
                channel_stats['moda'] = int(mode_result.mode[0])
                channel_stats['frecuencia_moda'] = int(mode_result.count[0])
            except:
                channel_stats['moda'] = int(np.argmax(np.bincount(channel_data)))
                channel_stats['frecuencia_moda'] = int(np.max(np.bincount(channel_data)))
            
            # Generar histograma
            hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
            channel_stats['histograma'] = {
                'valores': hist.tolist(),
                'bins': bins.tolist()
            }
            
            statistics[color_name] = channel_stats
        
        # Generar gráfico de histogramas
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        channel_names = ['rojo', 'verde', 'azul']
        
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            axes[i].hist(channels[name], bins=256, color=color, alpha=0.7, range=(0, 256))
            axes[i].set_title(f'Canal {name.capitalize()}')
            axes[i].set_xlabel('Valor de intensidad')
            axes[i].set_ylabel('Frecuencia')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir gráfico a base64 para mostrar en web
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        histogram_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        statistics['histograma_grafico'] = histogram_b64
        
        return statistics
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Página principal de la aplicación"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Manejar subida de archivos y análisis de imagen
    TODO: Añadir validación de tipo MIME
    TODO: Implementar procesamiento asíncrono para imágenes grandes
    """
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extraer metadatos
            metadata = extract_metadata(filepath)
            
            # Analizar canales RGB
            rgb_analysis = analyze_rgb_channels(filepath)
            
            # Preparar imagen para mostrar en web
            with Image.open(filepath) as img:
                # Redimensionar si es muy grande
                max_display_size = (800, 600)
                img.thumbnail(max_display_size, Image.Resampling.LANCZOS)
                
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Limpiar archivo temporal
            # TODO: Implementar limpieza automática de archivos antiguos
            os.remove(filepath)
            
            return render_template('results.html', 
                                 metadata=metadata,
                                 rgb_analysis=rgb_analysis,
                                 image_data=img_b64,
                                 filename=filename)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Error procesando la imagen: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Tipo de archivo no permitido. Use: PNG, JPG, JPEG, GIF, BMP, TIFF')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint para análisis de imágenes
    TODO: Implementar autenticación API
    TODO: Añadir límites de rate limiting
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        metadata = extract_metadata(filepath)
        rgb_analysis = analyze_rgb_channels(filepath)
        
        os.remove(filepath)
        
        return jsonify({
            'metadata': metadata,
            'rgb_analysis': rgb_analysis,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # TODO: Configurar logging apropiado
    # TODO: Usar gunicorn para producción
    app.run(host='0.0.0.0', port=5000, debug=True)