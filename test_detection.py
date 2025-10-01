#!/usr/bin/env python3
"""
Prueba de Detección de Personas en Imagen

Script simple para probar YOLOv8 con una imagen estática.
"""

from ultralytics import YOLO
import cv2
import os

def test_yolo_with_image(image_path='input/test_image.jpg'):
    """
    Prueba YOLOv8 con una imagen estática.
    
    Args:
        image_path (str): Ruta a la imagen de prueba
    """
    print("🔍 Probando YOLOv8 con imagen de prueba...")
    print("=" * 50)
    
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encontró la imagen {image_path}")
        return
    
    # Cargar modelo YOLOv8
    print("🔄 Cargando modelo YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("✅ Modelo cargado exitosamente")
    
    # Cargar imagen
    print(f"📸 Cargando imagen: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: No se pudo cargar la imagen")
        return
    
    height, width = img.shape[:2]
    print(f"📊 Resolución de imagen: {width}x{height}")
    
    # Ejecutar detección
    print("🚀 Ejecutando detección...")
    results = model(img, conf=0.3)  # Confianza baja para detectar más objetos
    
    # Analizar resultados
    result = results[0]
    
    total_detections = len(result.boxes) if result.boxes is not None else 0
    person_detections = 0
    
    print(f"\n📊 RESULTADOS DE DETECCIÓN:")
    print(f"🔍 Detecciones totales: {total_detections}")
    
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            print(f"  {i+1}. Clase: {class_name} | Confianza: {confidence:.3f} | ID: {class_id}")
            
            if class_id == 0:  # Persona
                person_detections += 1
    
    print(f"\n👥 Personas detectadas: {person_detections}")
    
    # Guardar imagen con detecciones
    if total_detections > 0:
        output_path = 'output/detection_result.jpg'
        os.makedirs('output', exist_ok=True)
        
        # Dibujar detecciones en la imagen
        annotated_img = result.plot()
        cv2.imwrite(output_path, annotated_img)
        print(f"💾 Imagen con detecciones guardada en: {output_path}")
    else:
        print("ℹ️  No se guardó imagen ya que no hubo detecciones")
    
    print("\n" + "=" * 50)
    
    if person_detections == 0:
        print("⚠️  NOTA: YOLOv8 no detectó personas en esta imagen.")
        print("   Esto puede deberse a:")
        print("   - Las siluetas no son lo suficientemente realistas")
        print("   - La confianza es muy baja")
        print("   - El modelo requiere características más detalladas")
        print("\n💡 RECOMENDACIÓN: Usa una imagen real con personas")
        print("   o prueba el sistema con un video real.")

if __name__ == "__main__":
    test_yolo_with_image()