#!/usr/bin/env python3
"""
Aplicación Flask para Detección de Armas
Módulos:
1. Detección en archivos (imágenes/videos) con drag & drop
2. Detección en tiempo real con webcam
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
from pathlib import Path
import sys
import cv2
import base64
import json
from datetime import datetime
import numpy as np

# Añadir directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent))
from weapon_detection_pipeline.pipeline import WeaponDetectionPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB máximo
app.config['UPLOAD_FOLDER'] = 'uploads/weapons'
app.config['RESULTS_FOLDER'] = 'results/weapons'

# Crear directorios necesarios
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(parents=True, exist_ok=True)

# Inicializar pipeline (carga los modelos una sola vez)
print("🔄 Inicializando pipeline de detección de armas...")
"""
Ruta del modelo:
  1) Usa WEAPON_MODEL_PATH si está definida (y apunta a un archivo existente)
  2) Si existe el directorio ./models, asume ./models/best_model.pth
  3) Fallback al path del repo: weapons_detector2/results_light/best_model.pth
"""
env_model_path = os.environ.get("WEAPON_MODEL_PATH")
candidate_paths = []
if env_model_path:
    candidate_paths.append(env_model_path)
# Si existe la carpeta models, probamos models/best_model.pth
if os.path.isdir("models"):
    candidate_paths.append("models/best_model.pth")
# Fallback del repositorio
candidate_paths.append("weapons_detector2/results_light/best_model.pth")

weapon_model_path = None
for p in candidate_paths:
    if p and os.path.isfile(p):
        weapon_model_path = p
        break

if not weapon_model_path:
    # Log de ayuda para diagnosticar en contenedor
    print("❌ No se encontró el archivo de modelo en las rutas candidatas:")
    for p in candidate_paths:
        print(f"   - {p} (exists_dir={os.path.isdir(os.path.dirname(p))} file={os.path.isfile(p)})")
    raise FileNotFoundError("No se encontró el modelo de armas. Configure WEAPON_MODEL_PATH o monte ./weapons_detector2/results_light en /app/models con best_model.pth")
else:
    print(f"✅ Usando modelo de armas en: {weapon_model_path}")

pipeline = WeaponDetectionPipeline(
    weapon_model_path=weapon_model_path,
    yolo_model_path="person_extraction/yolov8n.pt",
    confidence_threshold=0.5
)
print("✅ Pipeline listo")

# Variable global para control de webcam
webcam_active = False
current_webcam = None


@app.route('/')
def index():
    """Página principal - selector de módulos."""
    return render_template('weapon_detector_index.html')


@app.route('/file-detector')
def file_detector():
    """Módulo 1: Detección en archivos."""
    return render_template('weapon_file_detector.html')


@app.route('/realtime-detector')
def realtime_detector():
    """Módulo 2: Detección en tiempo real."""
    return render_template('weapon_realtime_detector.html')


@app.route('/api/detect-file', methods=['POST'])
def detect_file():
    """API para procesar archivos (imagen o video)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    # Obtener parámetros
    confidence = float(request.form.get('confidence', 0.5))
    pipeline.confidence_threshold = confidence
    
    # Guardar archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix
    input_filename = f"input_{timestamp}{file_ext}"
    input_path = Path(app.config['UPLOAD_FOLDER']) / input_filename
    file.save(str(input_path))
    
    try:
        # Determinar tipo de archivo
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        
        if file_ext.lower() in image_exts:
            # Procesar imagen
            image = cv2.imread(str(input_path))
            result_image, detections = pipeline.process_image(image)
            
            # Guardar resultado
            output_filename = f"result_{timestamp}.jpg"
            output_path = Path(app.config['RESULTS_FOLDER']) / output_filename
            cv2.imwrite(str(output_path), result_image)
            
            return jsonify({
                'success': True,
                'type': 'image',
                'detections': len(detections),
                'details': detections,
                'result_url': f'/results/weapons/{output_filename}'
            })
        
        elif file_ext.lower() in video_exts:
            # Procesar video
            output_filename = f"result_{timestamp}.mp4"
            output_path = Path(app.config['RESULTS_FOLDER']) / output_filename
            
            frame_skip = int(request.form.get('frame_skip', 2))
            _, frames_processed, total_detections = pipeline.process_video(
                str(input_path),
                str(output_path),
                frame_skip=frame_skip
            )
            
            return jsonify({
                'success': True,
                'type': 'video',
                'frames_processed': frames_processed,
                'detections': total_detections,
                'result_url': f'/results/weapons/{output_filename}'
            })
        
        else:
            return jsonify({'error': f'Formato no soportado: {file_ext}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Limpiar archivo de entrada
        if input_path.exists():
            input_path.unlink()


@app.route('/results/weapons/<filename>')
def serve_result(filename):
    """Sirve archivos de resultados."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


def generate_webcam_stream():
    """Generador para streaming de video con detección en tiempo real."""
    global webcam_active, current_webcam
    
    webcam_active = True
    current_webcam = cv2.VideoCapture(0)
    
    if not current_webcam.isOpened():
        webcam_active = False
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: No se pudo acceder a la webcam\r\n'
        return
    
    try:
        while webcam_active:
            ret, frame = current_webcam.read()
            if not ret:
                break
            
            # Procesar frame
            annotated_frame, detections = pipeline.process_image(frame)
            
            # Agregar información en pantalla
            info_text = f"Detecciones: {len(detections)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Codificar frame a JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            # Enviar frame en formato multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        if current_webcam:
            current_webcam.release()
        webcam_active = False


@app.route('/api/webcam/stream')
def webcam_stream():
    """Stream de video de la webcam con detección."""
    return Response(
        generate_webcam_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Inicia la detección en webcam."""
    global webcam_active
    
    data = request.get_json()
    confidence = data.get('confidence', 0.5)
    pipeline.confidence_threshold = confidence
    
    return jsonify({'success': True, 'message': 'Webcam iniciada'})


@app.route('/api/webcam/stop', methods=['POST'])
def stop_webcam():
    """Detiene la detección en webcam."""
    global webcam_active, current_webcam
    
    webcam_active = False
    if current_webcam:
        current_webcam.release()
        current_webcam = None
    
    return jsonify({'success': True, 'message': 'Webcam detenida'})


@app.route('/api/webcam/capture', methods=['POST'])
def capture_frame():
    """Captura el frame actual de la webcam."""
    global current_webcam
    
    if not current_webcam or not current_webcam.isOpened():
        return jsonify({'error': 'Webcam no activa'}), 400
    
    ret, frame = current_webcam.read()
    if not ret:
        return jsonify({'error': 'No se pudo capturar el frame'}), 500
    
    # Procesar y guardar
    annotated_frame, detections = pipeline.process_image(frame)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"capture_{timestamp}.jpg"
    output_path = Path(app.config['RESULTS_FOLDER']) / output_filename
    cv2.imwrite(str(output_path), annotated_frame)
    
    return jsonify({
        'success': True,
        'detections': len(detections),
        'details': detections,
        'image_url': f'/results/weapons/{output_filename}'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
