#!/usr/bin/env python3
"""
Generador de Video de Prueba

Crea un video simple con formas geométricas para probar el sistema de detección.
"""

import cv2
import numpy as np
import os

def create_test_video(output_path='input/test_video.mp4', duration=5, fps=30):
    """
    Crea un video de prueba con formas geométricas.
    
    Args:
        output_path (str): Ruta donde guardar el video
        duration (int): Duración en segundos
        fps (int): Frames por segundo
    """
    # Configuración del video
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"🎬 Creando video de prueba...")
    print(f"   - Resolución: {width}x{height}")
    print(f"   - Duración: {duration}s ({total_frames} frames)")
    print(f"   - FPS: {fps}")
    
    for frame_num in range(total_frames):
        # Crear frame con fondo azul
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Agregar algunas "personas" simuladas (rectángulos y círculos)
        progress = frame_num / total_frames
        
        # Persona 1: Rectángulo que se mueve de izquierda a derecha
        x1 = int(50 + progress * (width - 150))
        y1 = height - 200
        cv2.rectangle(frame, (x1, y1), (x1 + 60, y1 + 150), (255, 200, 100), -1)
        cv2.circle(frame, (x1 + 30, y1 - 20), 25, (255, 220, 180), -1)  # Cabeza
        
        # Persona 2: Círculo que se mueve verticalmente
        x2 = width - 100
        y2 = int(100 + progress * 200)
        cv2.rectangle(frame, (x2, y2), (x2 + 50, y2 + 120), (100, 255, 100), -1)
        cv2.circle(frame, (x2 + 25, y2 - 15), 20, (150, 255, 150), -1)  # Cabeza
        
        # Persona 3: Estática en el centro
        x3, y3 = width // 2 - 25, height // 2
        cv2.rectangle(frame, (x3, y3), (x3 + 50, y3 + 100), (100, 100, 255), -1)
        cv2.circle(frame, (x3 + 25, y3 - 15), 18, (150, 150, 255), -1)  # Cabeza
        
        # Agregar texto informativo
        cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Video de Prueba - Detección de Personas", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
        # Mostrar progreso
        if frame_num % 30 == 0:
            progress_percent = (frame_num / total_frames) * 100
            print(f"   Progreso: {progress_percent:.1f}%")
    
    out.release()
    print(f"✅ Video creado exitosamente: {output_path}")
    print(f"📊 Tamaño del archivo: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    create_test_video()