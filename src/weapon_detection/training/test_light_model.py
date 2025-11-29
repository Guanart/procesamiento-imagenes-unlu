#!/usr/bin/env python3
"""
Test Model - Evalúa el modelo entrenado en el conjunto de test

Calcula métricas de precisión (mAP) en el conjunto de test separado
y guarda imágenes con detecciones visualizadas.

Autor: Procesamiento de Imágenes - UNLU
Fecha: Noviembre 2025
"""

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_MODEL_PATH = "results_light/checkpoint_epoch_final.pth"
DEFAULT_TEST_IMAGES_DIR = "dataset_testing/images"
DEFAULT_TEST_XML_DIR = "dataset_testing/xmls"
DEFAULT_OUTPUT_DIR = "test_results"
DEFAULT_CONFIDENCE = 0.5

# Clases (deben coincidir con las usadas en el entrenamiento)
CLASSES = ["__background__", "knife", "pistol"]
CLASS_MAP = {"knife": 1, "pistol": 2}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_model(num_classes):
    """Crea el modelo Faster R-CNN con backbone MobileNetV3-Large FPN."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    return model


def parse_voc_xml(xml_path):
    """Parse XML de Pascal VOC y retorna lista de boxes."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls_name = name_el.text.strip().lower()
        if cls_name not in CLASS_MAP:
            continue
        bbox_el = obj.find("bndbox")
        if bbox_el is None:
            continue
        try:
            xmin = int(float(bbox_el.find("xmin").text or "0"))
            ymin = int(float(bbox_el.find("ymin").text or "0"))
            xmax = int(float(bbox_el.find("xmax").text or "0"))
            ymax = int(float(bbox_el.find("ymax").text or "0"))
        except (ValueError, AttributeError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append({
            "boxes": torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32),
            "labels": torch.tensor([CLASS_MAP[cls_name]], dtype=torch.int64)
        })
    return boxes


def find_image_for_xml(xml_path, images_dir):
    """Encuentra la imagen correspondiente al XML."""
    for ext in VALID_EXTS:
        image_path = images_dir / f"{xml_path.stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def test_model(model_path, test_images_dir, test_xml_dir, output_dir, confidence_threshold, save_images=True):
    """Evalúa el modelo en el conjunto de test y calcula métricas."""
    print(f"🔧 Usando dispositivo: {DEVICE}")

    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en '{model_path}'.")
        return

    test_images_dir = Path(test_images_dir)
    test_xml_dir = Path(test_xml_dir)
    output_dir = Path(output_dir)

    if not test_xml_dir.exists():
        print(f"❌ Error: Directorio de XMLs no encontrado: {test_xml_dir}")
        return

    print(f"📥 Cargando modelo desde '{model_path}'...")
    model = get_model(num_classes=len(CLASSES))
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modelo cargado desde época {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Modelo cargado (formato antiguo)")
    
    model.to(DEVICE)
    model.eval()

    # Obtener lista de XMLs de test
    xml_files = sorted(test_xml_dir.glob('*.xml'))
    
    if not xml_files:
        print(f"❌ No se encontraron archivos XML en {test_xml_dir}")
        return

    print(f"🧪 Evaluando en {len(xml_files)} imágenes de test...")
    
    # Crear directorio de salida si se guardan imágenes
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar métricas
    metric = MeanAveragePrecision(box_format="xyxy")
    
    total_detections = 0
    total_ground_truth = 0
    failed_images = 0

    # Procesar cada imagen
    for xml_path in tqdm(xml_files, desc="Evaluando", unit="img"):
        try:
            # Encontrar imagen correspondiente
            image_path = find_image_for_xml(xml_path, test_images_dir)
            if image_path is None:
                print(f"⚠️  Imagen no encontrada para: {xml_path.name}")
                failed_images += 1
                continue

            # Cargar imagen
            image = Image.open(image_path).convert("RGB")
            
            # Preparar tensor para inferencia
            image_tensor = F.to_tensor(image).to(DEVICE)
            
            # Inferencia
            with torch.no_grad():
                predictions = model([image_tensor])
            
            # Parse ground truth
            gt_boxes = parse_voc_xml(xml_path)
            if not gt_boxes:
                continue
            
            # Preparar targets para métricas
            target = {
                "boxes": torch.cat([b["boxes"] for b in gt_boxes]),
                "labels": torch.cat([b["labels"] for b in gt_boxes])
            }
            
            # Actualizar contadores
            total_ground_truth += len(target["labels"])
            pred = predictions[0]
            high_conf_mask = pred["scores"] >= confidence_threshold
            total_detections += high_conf_mask.sum().item()
            
            # Actualizar métricas
            metric.update([pred], [target])
            
            # Guardar imagen con detecciones si se requiere
            if save_images:
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                
                # Dibujar predicciones
                for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                    if score >= confidence_threshold:
                        class_name = CLASSES[label.item()]
                        draw.rectangle(box.tolist(), outline="red", width=3)
                        text = f"{class_name}: {score:.2f}"
                        
                        try:
                            bbox = draw.textbbox((box[0], box[1]), text, font=font)
                            draw.rectangle(bbox, fill="red")
                            draw.text((box[0], box[1]), text, fill="white", font=font)
                        except AttributeError:
                            draw.text((box[0], box[1] - 10), text, fill="red", font=font)
                
                # Guardar imagen
                output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
                image.save(output_path)
        
        except Exception as e:
            print(f"❌ Error procesando {xml_path.name}: {e}")
            failed_images += 1

    # Calcular métricas finales
    print("\n�� Calculando métricas...")
    stats = metric.compute()
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("=" * 70)
    print(f"🧪 Imágenes de test: {len(xml_files)}")
    print(f"✅ Procesadas exitosamente: {len(xml_files) - failed_images}")
    print(f"❌ Errores: {failed_images}")
    print(f"🎯 Ground truth total: {total_ground_truth} objetos")
    print(f"🔍 Detecciones (conf >= {confidence_threshold}): {total_detections}")
    print("\n📈 Métricas de Precisión:")
    print(f"   mAP: {stats['map']:.4f}")
    print(f"   mAP@50: {stats['map_50']:.4f}")
    print(f"   mAP@75: {stats['map_75']:.4f}")
    print(f"   mAP (small): {stats['map_small']:.4f}")
    print(f"   mAP (medium): {stats['map_medium']:.4f}")
    print(f"   mAP (large): {stats['map_large']:.4f}")
    
    if save_images:
        print(f"\n💾 Imágenes con detecciones guardadas en: {output_dir}")
    
    print("=" * 70)
    
    # Guardar métricas en JSON
    results = {
        "model_path": str(model_path),
        "test_images": len(xml_files),
        "processed": len(xml_files) - failed_images,
        "failed": failed_images,
        "confidence_threshold": confidence_threshold,
        "total_ground_truth": total_ground_truth,
        "total_detections": total_detections,
        "metrics": {
            "map": float(stats["map"]),
            "map_50": float(stats["map_50"]),
            "map_75": float(stats["map_75"]),
            "map_small": float(stats["map_small"]),
            "map_medium": float(stats["map_medium"]),
            "map_large": float(stats["map_large"])
        }
    }
    
    results_path = output_dir / "test_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Métricas guardadas en: {results_path}")
    
    return results


def get_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evalúa el modelo Faster R-CNN en el conjunto de test."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al modelo entrenado (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default=DEFAULT_TEST_IMAGES_DIR,
        help=f"Directorio con imágenes de test (default: {DEFAULT_TEST_IMAGES_DIR})",
    )
    parser.add_argument(
        "--test-xml-dir",
        type=str,
        default=DEFAULT_TEST_XML_DIR,
        help=f"Directorio con XMLs de test (default: {DEFAULT_TEST_XML_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Umbral de confianza (0.0-1.0) (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="No guardar imágenes con detecciones (solo calcular métricas)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_model(
        args.model,
        args.test_images_dir,
        args.test_xml_dir,
        args.output,
        args.confidence,
        save_images=not args.no_save_images
    )
