#!/usr/bin/env python3
"""
Generador de Imagen de Prueba con Personas

Crea una imagen que contenga siluetas más realistas de personas
para probar el sistema de detección.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def create_realistic_person_silhouette(width=60, height=150):
    """
    Crea una silueta de persona más realista usando PIL.
    
    Args:
        width (int): Ancho de la silueta
        height (int): Alto de la silueta
        
    Returns:
        np.array: Imagen de la silueta en formato OpenCV
    """
    # Crear imagen PIL
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Cabeza (círculo)
    head_radius = width // 6
    head_center = (width // 2, head_radius + 5)
    draw.ellipse([
        head_center[0] - head_radius, head_center[1] - head_radius,
        head_center[0] + head_radius, head_center[1] + head_radius
    ], fill=(255, 220, 180, 255))
    
    # Torso (rectángulo con esquinas redondeadas)
    torso_top = head_center[1] + head_radius + 5
    torso_height = height * 0.4
    torso_width = width * 0.6
    torso_left = (width - torso_width) // 2
    
    draw.rectangle([
        torso_left, torso_top,
        torso_left + torso_width, torso_top + torso_height
    ], fill=(100, 150, 200, 255))
    
    # Brazos
    arm_width = width * 0.15
    arm_height = torso_height * 0.8
    
    # Brazo izquierdo
    draw.rectangle([
        torso_left - arm_width, torso_top + 10,
        torso_left, torso_top + 10 + arm_height
    ], fill=(100, 150, 200, 255))
    
    # Brazo derecho
    draw.rectangle([
        torso_left + torso_width, torso_top + 10,
        torso_left + torso_width + arm_width, torso_top + 10 + arm_height
    ], fill=(100, 150, 200, 255))
    
    # Piernas
    leg_width = width * 0.25
    leg_height = height - (torso_top + torso_height)
    leg_gap = width * 0.05
    
    # Pierna izquierda
    draw.rectangle([
        width // 2 - leg_gap - leg_width, torso_top + torso_height,
        width // 2 - leg_gap, torso_top + torso_height + leg_height
    ], fill=(50, 50, 150, 255))
    
    # Pierna derecha
    draw.rectangle([
        width // 2 + leg_gap, torso_top + torso_height,
        width // 2 + leg_gap + leg_width, torso_top + torso_height + leg_height
    ], fill=(50, 50, 150, 255))
    
    # Convertir a formato OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    return img_cv

def create_test_image_with_persons(output_path='input/test_image.jpg'):
    """
    Crea una imagen de prueba con siluetas realistas de personas.
    
    Args:
        output_path (str): Ruta donde guardar la imagen
    """
    # Configuración de la imagen
    width, height = 800, 600
    
    # Crear imagen base con fondo
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Fondo gris claro
    
    # Agregar un "suelo"
    cv2.rectangle(img, (0, height - 50), (width, height), (100, 100, 100), -1)
    
    print("🖼️ Creando imagen de prueba con personas realistas...")
    
    # Crear diferentes personas
    persons_data = [
        {"pos": (100, height - 200), "size": (60, 150), "color_offset": 0},
        {"pos": (300, height - 180), "size": (55, 140), "color_offset": 30},
        {"pos": (500, height - 190), "size": (65, 160), "color_offset": 60},
        {"pos": (650, height - 175), "size": (58, 145), "color_offset": 90},
    ]
    
    for i, person_data in enumerate(persons_data):
        # Crear silueta
        person_img = create_realistic_person_silhouette(
            person_data["size"][0], 
            person_data["size"][1]
        )
        
        # Posición donde colocar la persona
        x, y = person_data["pos"]
        h, w = person_img.shape[:2]
        
        # Asegurar que la persona esté dentro de la imagen
        if x + w <= width and y + h <= height and x >= 0 and y >= 0:
            # Crear máscara para composición
            mask = np.any(person_img != [0, 0, 0], axis=-1)
            img[y:y+h, x:x+w][mask] = person_img[mask]
    
    # Agregar texto descriptivo
    cv2.putText(img, "Imagen de Prueba - Detección de Personas", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, f"Contiene {len(persons_data)} personas simuladas", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    
    print(f"✅ Imagen creada exitosamente: {output_path}")
    print(f"📊 Resolución: {width}x{height}")
    print(f"👥 Personas incluidas: {len(persons_data)}")
    print(f"📄 Tamaño del archivo: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    create_test_image_with_persons()