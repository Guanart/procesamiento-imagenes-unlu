#!/usr/bin/env python3
"""
Script de inferencia para probar el modelo Faster R-CNN entrenado en detección de armas.
"""

import torch
import torchvision
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_model(model_path, num_classes):
    """Carga el modelo Faster R-CNN entrenado."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(model, image_path, device, confidence_threshold=0.5):
    """Hace predicción en una imagen."""
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        predictions = model(image_tensor)

    # Debug: imprimir predicciones crudas
    print(f"Predicción cruda: {predictions}")
    
    # Procesar resultados
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    
    print(f"Boxes shape: {pred_boxes.shape}")
    print(f"Scores: {pred_scores}")
    print(f"Labels: {pred_labels}")

    # Filtrar por confidence
    keep = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    
    print(f"Detecciones después del filtro: {len(pred_boxes)}")

    return pred_boxes, pred_scores, pred_labels, image

def visualize_prediction(image, boxes, scores, labels, class_names, save_path=None):
    """Visualiza las predicciones en la imagen."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Colores para diferentes clases
    colors = ['red', 'blue', 'green']

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        class_name = class_names.get(label, f'class_{label}')
        color = colors[label % len(colors)]

        # Dibujar bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Etiqueta
        ax.text(x1, y1-5, f'{class_name}: {score:.2f}',
               color=color, fontsize=12, weight='bold',
               bbox=dict(facecolor='white', alpha=0.8))

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Imagen guardada en: {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    # Configuración
    model_path = "results_real/best_model.pth"
    classes_path = "results_real/classes.json"
    test_image_path = "weapons_detector/dataset/images/knife_1.jpg"  # Primera imagen del dataset

    device = torch.device('cpu')
    confidence_threshold = 0.1  # Threshold más bajo para ver todas las detecciones

    print("🚀 Probando modelo Faster R-CNN...")
    print(f"Modelo: {model_path}")
    print(f"Imagen de prueba: {test_image_path}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {confidence_threshold}")

    # Verificar archivos
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    if not os.path.exists(classes_path):
        print(f"❌ Archivo de clases no encontrado: {classes_path}")
        return

    if not os.path.exists(test_image_path):
        print(f"❌ Imagen de prueba no encontrada: {test_image_path}")
        return

    # Cargar clases
    with open(classes_path, 'r') as f:
        classes_data = json.load(f)
        class_names = {v: k for k, v in classes_data['classes'].items()}
        num_classes = len(classes_data['classes']) + 1  # + background

    print(f"Clases: {class_names}")
    print(f"Número de clases: {num_classes}")

    # Cargar modelo
    model = load_model(model_path, num_classes)
    model.to(device)

    # Hacer predicción
    boxes, scores, labels, image = predict_image(model, test_image_path, device, confidence_threshold)

    print("\nResultados de la prediccion:")
    print(f"Numero de detecciones: {len(boxes)}")

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_name = class_names.get(label, f'class_{label}')
        print(f"Deteccion {i+1}: {class_name} - Confianza: {score:.2f} - Box: {box}")

    # Visualizar
    output_path = "results_real/test_prediction.png"
    visualize_prediction(image, boxes, scores, labels, class_names, output_path)

    print("\nPrueba completada!")
    print(f"Imagen con predicciones guardada en: {output_path}")

if __name__ == "__main__":
    main()