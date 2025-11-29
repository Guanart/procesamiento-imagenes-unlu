#!/usr/bin/env python3
"""
Video Processor - Stage 1: Detección y Extracción de Personas

Este script implementa el primer stage de un sistema de detección de riesgo.
Procesa un video frame por frame y extrae imágenes de todas las personas detectadas
utilizando YOLOv8 pre-entrenado en el dataset COCO.

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Septiembre 2025
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Tuple

# Configuración de rutas por defecto
DEFAULT_VIDEO_PATH = 'input/test_video.mp4'  # Placeholder para el video de entrada
DEFAULT_OUTPUT_DIR = 'output/cropped_persons'  # Directorio de salida para personas extraídas

# Variables globales que se actualizarán con argumentos
VIDEO_PATH = DEFAULT_VIDEO_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# ID de la clase 'person' en el dataset COCO
PERSON_CLASS_ID = 0

def setup_directories() -> None:
    """
    Crea los directorios necesarios si no existen.
    
    Crea:
    - Directorio de entrada (input/)
    - Directorio de salida (output/cropped_persons/)
    """
    Path('input').mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Directorios configurados:")
    print(f"   - Entrada: input/")
    print(f"   - Salida: {OUTPUT_DIR}/")

def load_yolo_model() -> YOLO:
    """
    Carga el modelo YOLOv8n pre-entrenado en COCO.
    
    Returns:
        YOLO: Modelo YOLOv8n listo para inferencia
        
    Notes:
        - Usa YOLOv8n (nano) por su eficiencia y velocidad
        - El modelo está pre-entrenado en COCO dataset
        - Incluye la clase 'person' con ID=0
    """
    print("🔄 Cargando modelo YOLOv8n...")
    model = YOLO('yolov8n.pt')  # Descarga automáticamente si no existe
    print("✅ Modelo YOLOv8n cargado exitosamente")
    print(f"   - Arquitectura: CNN de una sola etapa (single-stage)")
    print(f"   - Dataset de entrenamiento: COCO (80 clases)")
    print(f"   - Clase objetivo: 'person' (ID: {PERSON_CLASS_ID})")
    return model

def extract_person_crops(frame: cv2.Mat, results, processed_frame_number: int) -> int:
    """
    Extrae y guarda recortes de todas las personas detectadas en un frame.
    
    Args:
        frame: Frame del video (imagen BGR)
        results: Resultados de la inferencia YOLOv8
        processed_frame_number: Número secuencial del frame procesado (1, 2, 3...)
        
    Returns:
        int: Número de personas detectadas y guardadas
    """
    persons_detected = 0
    
    # Procesar cada detección en el frame
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Filtrar solo detecciones de la clase 'person' (ID: 0)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == PERSON_CLASS_ID and confidence > 0.5:  # Umbral de confianza
                    # Obtener coordenadas de la bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Asegurar que las coordenadas estén dentro de los límites del frame
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Recortar la región de la persona
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:  # Verificar que el recorte no esté vacío
                        # Generar nombre del archivo con contador secuencial
                        filename = f"frame{processed_frame_number}_person_{i+1}_conf_{confidence:.2f}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        # Guardar el recorte
                        cv2.imwrite(filepath, person_crop)
                        persons_detected += 1
                        
                        print(f"   💾 Guardado: {filename} (conf: {confidence:.2f})")
    
    return persons_detected

def process_video(video_path: str, model: YOLO) -> Tuple[int, int]:
    """
    Procesa un video completo y extrae todas las personas detectadas.
    
    Args:
        video_path: Ruta al archivo de video
        model: Modelo YOLOv8 cargado
        
    Returns:
        Tuple[int, int]: (total_frames_procesados, total_personas_extraídas)
    """
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"❌ Error: No se pudo abrir el video {video_path}")
    
    # Obtener información del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calcular intervalo para 1 frame por segundo
    frame_interval = max(1, int(fps))  # Saltar frames para obtener ~1 fps
    frames_to_process = total_frames // frame_interval
    
    print(f"\n📹 Información del video:")
    print(f"   - Frames totales: {total_frames}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Duración: {duration:.2f} segundos")
    print(f"   - Intervalo de muestreo: 1 frame cada {frame_interval} frames")
    print(f"   - Frames a procesar: {frames_to_process}")
    print(f"\n🚀 Iniciando procesamiento...")
    
    frame_count = 0
    processed_frames = 0
    total_persons = 0
    
    try:
        while True:
            # Leer frame por frame
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video
            
            frame_count += 1
            
            # Procesar solo cada frame_interval frames (aproximadamente 1 por segundo)
            if frame_count % frame_interval == 0:
                processed_frames += 1
                
                # Mostrar progreso
                if processed_frames % 5 == 0 or processed_frames == 1:
                    progress = (frame_count / total_frames) * 100
                    print(f"\n🔍 Frame {frame_count}/{total_frames} ({progress:.1f}%) - Procesado #{processed_frames}")
                
                # Realizar inferencia con YOLOv8
                # verbose=False para reducir output del modelo
                results = model(frame, verbose=False)
                
                # Extraer y guardar personas detectadas (usando contador secuencial)
                persons_in_frame = extract_person_crops(frame, results, processed_frames)
                total_persons += persons_in_frame
            
    except KeyboardInterrupt:
        print("\n⚠️  Procesamiento interrumpido por el usuario")
    finally:
        cap.release()
    
    return processed_frames, total_persons

def main():
    """
    Función principal del programa.
    
    Flujo de trabajo:
    1. Configurar directorios
    2. Cargar modelo YOLOv8
    3. Procesar video frame por frame
    4. Detectar personas usando YOLO
    5. Extraer y guardar recortes de personas
    6. Mostrar estadísticas finales
    """
    print("🎯 Stage 1: Detección y Extracción de Personas")
    print("=" * 50)
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Extrae personas de un video usando YOLOv8"
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help=f"Ruta al video de entrada (default: {DEFAULT_VIDEO_PATH})"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Actualizar rutas con argumentos
    global VIDEO_PATH, OUTPUT_DIR
    VIDEO_PATH = args.video
    OUTPUT_DIR = args.output
    
    try:
        # 1. Configurar directorios
        setup_directories()
        
        # 2. Verificar que existe el video
        if not os.path.exists(VIDEO_PATH):
            print(f"\n❌ Error: Video no encontrado en {VIDEO_PATH}")
            print(f"   Coloca tu video en la carpeta 'input/' o especifica la ruta con --video")
            return
        
        # 3. Cargar modelo YOLOv8
        model = load_yolo_model()
        
        # 4. Procesar video
        frames_processed, total_persons = process_video(VIDEO_PATH, model)
        
        # 5. Mostrar estadísticas finales
        print("\n" + "=" * 50)
        print("📊 RESUMEN DEL PROCESAMIENTO")
        print("=" * 50)
        print(f"✅ Frames procesados: {frames_processed}")
        print(f"👥 Personas extraídas: {total_persons}")
        print(f"📁 Imágenes guardadas en: {OUTPUT_DIR}/")
        print(f"\n🎯 Stage 1 completado exitosamente!")
        print(f"💡 Las imágenes extraídas están listas para el Stage 2")
        print(f"   (Fine-tuning para detección de armas)")
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())