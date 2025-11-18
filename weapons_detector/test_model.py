#!/usr/bin/env python3
"""
Script para probar el modelo Faster R-CNN entrenado.
Realiza predicciones en imágenes y muestra/guarda los resultados.

Uso:
    python test_model.py --model results_frcnn/best_model.pth --image path/to/image.jpg
    python test_model.py --model results_frcnn/best_model.pth --image-dir dataset/images --output predictions/
"""

import argparse
import json
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple

# Mapa de clases (debe coincidir con el entrenamiento)
CLASS_MAP = {"knife": 1, "pistol": 2}
CLASSES = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP) + 1  # + background

# Colores para visualización
COLORS = {
    'knife': (255, 0, 0),    # Rojo
    'pistol': (0, 255, 0),   # Verde
}

def create_model(num_classes: int, model_path: str, device: torch.device):
    """Carga el modelo entrenado."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Cargar pesos
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def load_image(image_path: Path) -> Tuple[torch.Tensor, Image.Image]:
    """Carga y preprocesa una imagen."""
    # Cargar imagen original para visualización
    img_pil = Image.open(image_path).convert('RGB')
    
    # Convertir a tensor y normalizar
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil)
    
    return img_tensor, img_pil

def predict(model: torch.nn.Module, image_tensor: torch.Tensor, 
           device: torch.device, confidence_threshold: float = 0.5) -> Dict:
    """Realiza predicción en una imagen."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])[0]
    
    # Filtrar por confianza
    scores = predictions['scores'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    mask = scores >= confidence_threshold
    
    return {
        'boxes': boxes[mask],
        'scores': scores[mask],
        'labels': labels[mask]
    }

def draw_predictions(image: Image.Image, predictions: Dict, 
                    save_path: Path = None, show: bool = True) -> Image.Image:
    """Dibuja las predicciones en la imagen."""
    draw = ImageDraw.Draw(image)
    
    # Intentar cargar fuente, usar default si falla
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        xmin, ymin, xmax, ymax = box
        class_name = CLASSES.get(label, 'unknown')
        color = COLORS.get(class_name, (255, 255, 0))
        
        # Dibujar caja
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # Dibujar etiqueta con fondo
        label_text = f"{class_name}: {score:.2f}"
        
        # Calcular tamaño del texto
        bbox = draw.textbbox((xmin, ymin), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Fondo para el texto
        draw.rectangle(
            [xmin, ymin - text_height - 4, xmin + text_width + 4, ymin],
            fill=color
        )
        
        # Texto
        draw.text((xmin + 2, ymin - text_height - 2), label_text, 
                 fill=(255, 255, 255), font=font)
    
    # Guardar si se especifica
    if save_path:
        image.save(save_path)
        print(f"💾 Imagen guardada en: {save_path}")
    
    # Mostrar si se requiere
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Detecciones: {len(predictions['boxes'])}")
        plt.tight_layout()
        plt.show()
    
    return image

def print_predictions(predictions: Dict, image_path: Path):
    """Imprime resumen de predicciones."""
    print(f"\n{'='*60}")
    print(f"📸 Imagen: {image_path.name}")
    print(f"{'='*60}")
    
    if len(predictions['boxes']) == 0:
        print("❌ No se detectaron armas")
        return
    
    print(f"✅ Se detectaron {len(predictions['boxes'])} objeto(s):\n")
    
    for i, (box, score, label) in enumerate(zip(predictions['boxes'], 
                                                 predictions['scores'], 
                                                 predictions['labels']), 1):
        class_name = CLASSES.get(label, 'unknown')
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        
        print(f"  {i}. {class_name.upper()}")
        print(f"     Confianza: {score:.3f} ({score*100:.1f}%)")
        print(f"     Posición: ({xmin:.0f}, {ymin:.0f})")
        print(f"     Tamaño: {w:.0f}x{h:.0f} px")
        print()

def process_single_image(model, image_path: Path, output_dir: Path, 
                        device: torch.device, confidence: float, show: bool):
    """Procesa una sola imagen."""
    print(f"\n🔍 Procesando: {image_path.name}")
    
    # Cargar imagen
    img_tensor, img_pil = load_image(image_path)
    
    # Predecir
    predictions = predict(model, img_tensor, device, confidence)
    
    # Mostrar resultados
    print_predictions(predictions, image_path)
    
    # Dibujar y guardar
    output_path = output_dir / f"pred_{image_path.name}" if output_dir else None
    draw_predictions(img_pil.copy(), predictions, output_path, show)

def process_directory(model, image_dir: Path, output_dir: Path, 
                     device: torch.device, confidence: float, max_images: int):
    """Procesa un directorio de imágenes."""
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en {image_dir}")
        return
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n📂 Procesando {len(image_files)} imágenes...")
    
    results = []
    for img_path in image_files:
        img_tensor, img_pil = load_image(img_path)
        predictions = predict(model, img_tensor, device, confidence)
        
        # Guardar resultado
        output_path = output_dir / f"pred_{img_path.name}"
        draw_predictions(img_pil.copy(), predictions, output_path, show=False)
        
        results.append({
            'image': img_path.name,
            'detections': len(predictions['boxes']),
            'objects': [
                {
                    'class': CLASSES.get(label, 'unknown'),
                    'confidence': float(score),
                    'box': box.tolist()
                }
                for box, score, label in zip(predictions['boxes'], 
                                             predictions['scores'], 
                                             predictions['labels'])
            ]
        })
        
        print(f"  ✅ {img_path.name}: {len(predictions['boxes'])} detección(es)")
    
    # Guardar resumen JSON
    json_path = output_dir / 'predictions_summary.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resumen guardado en: {json_path}")
    
    # Estadísticas
    total_detections = sum(r['detections'] for r in results)
    images_with_detections = sum(1 for r in results if r['detections'] > 0)
    
    print(f"\n{'='*60}")
    print(f"📊 ESTADÍSTICAS")
    print(f"{'='*60}")
    print(f"Total de imágenes procesadas: {len(results)}")
    print(f"Imágenes con detecciones: {images_with_detections} ({images_with_detections/len(results)*100:.1f}%)")
    print(f"Total de detecciones: {total_detections}")
    print(f"Promedio detecciones/imagen: {total_detections/len(results):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Probar modelo Faster R-CNN')
    parser.add_argument('--model', required=True, help='Ruta al modelo (.pth)')
    parser.add_argument('--image', type=str, help='Imagen individual a procesar')
    parser.add_argument('--image-dir', type=str, help='Directorio con imágenes')
    parser.add_argument('--output', type=str, default='predictions', 
                       help='Directorio de salida')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Umbral de confianza (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Dispositivo (cuda/cpu)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Máximo de imágenes a procesar (dir mode)')
    parser.add_argument('--no-show', action='store_true',
                       help='No mostrar imágenes (solo guardar)')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("🎯 TEST DE MODELO FASTER R-CNN")
    print("="*60)
    print(f"Modelo: {args.model}")
    print(f"Dispositivo: {device}")
    print(f"Umbral confianza: {args.confidence}")
    print("="*60)
    
    # Verificar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        return 1
    
    # Cargar modelo
    print("\n🤖 Cargando modelo...")
    model = create_model(NUM_CLASSES, model_path, device)
    print("✅ Modelo cargado exitosamente")
    
    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar imagen(es)
    if args.image:
        # Modo imagen individual
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Error: Imagen no encontrada en {image_path}")
            return 1
        
        process_single_image(model, image_path, output_dir, device, 
                           args.confidence, not args.no_show)
    
    elif args.image_dir:
        # Modo directorio
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"❌ Error: Directorio no encontrado en {image_dir}")
            return 1
        
        process_directory(model, image_dir, output_dir, device, 
                         args.confidence, args.max_images)
    
    else:
        print("❌ Error: Debes especificar --image o --image-dir")
        return 1
    
    print(f"\n✅ Procesamiento completado")
    print(f"📁 Resultados guardados en: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
