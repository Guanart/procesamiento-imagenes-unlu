#!/usr/bin/env python3
"""
Entrenamiento Faster R-CNN Ligero (MobileNetV3) para detección de armas usando anotaciones Pascal VOC (.xml).
Versión optimizada para ALTA VELOCIDAD:
- Redimensiona imágenes a un tamaño fijo (ej. 800x800) para acelerar drásticamente el entrenamiento.
- Ajusta las coordenadas de las bounding boxes automáticamente.
- Carga de datos bajo demanda para bajo consumo de RAM.
- Métricas de evaluación profesional (mAP).
"""

import argparse
import time
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import concurrent.futures

# Configuración ROCm para AMD (reduce fragmentación de memoria)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Importar el enhancer
from image_enhancer import ImageEnhancer
import psutil

# --- CONFIGURACIÓN GLOBAL ---
# Mapa de clases (background implícito): labels deben iniciar en 1
CLASS_MAP = {"knife": 1, "pistol": 2}
NUM_CLASSES = len(CLASS_MAP) + 1  # + background
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Tamaño al que se redimensionarán todas las imágenes. ¡La clave para la velocidad!
RESIZE_TO = (320, 320)


def parse_voc_xml(xml_path: Path) -> Tuple[str, List[Tuple[int, int, int, int, int]]]:
    """
    Devuelve (filename, lista de cajas: (label,xmin,ymin,xmax,ymax))
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError:
        return "", []

    filename_el = root.find("filename")
    if filename_el is None or not filename_el.text:
        filename = xml_path.stem + ".jpg"
    else:
        filename = filename_el.text.strip()
        if "." not in filename:
            filename = filename + ".jpg"

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
        boxes.append((CLASS_MAP[cls_name], xmin, ymin, xmax, ymax))
    return filename, boxes


class WeaponDetectionDataset(Dataset):
    """
    Dataset que carga imágenes, las redimensiona y ajusta las bounding boxes al vuelo.
    Aplica mejoramiento de imagen (CLAHE, denoising, sharpening) antes de procesar.
    """

    def __init__(
        self, images_dir: Path, xml_files: List[Path], resize_to: Tuple[int, int], use_enhancement: bool = True
    ):
        self.images_dir = images_dir
        self.resize_to = resize_to
        self.samples = []
        self.use_enhancement = use_enhancement
        
        # Inicializar el enhancer
        if self.use_enhancement:
            self.enhancer = ImageEnhancer()
            print("✅ ImageEnhancer inicializado para mejorar calidad de imágenes")

        print(f"📦 Parseando {len(xml_files)} archivos XML (usando hilos para optimizar I/O de Drive)...")
        
        # CAMBIO: Usamos ThreadPoolExecutor en lugar de ProcessPoolExecutor.
        # En Google Colab con Drive, el cuello de botella es la latencia de red/disco (I/O), no la CPU.
        # Usamos más workers (16) para paralelizar las peticiones de lectura y ocultar la latencia.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(parse_voc_xml, xml_files), total=len(xml_files), desc="Parseando XMLs", leave=False))

        # Procesamos los resultados (resolución de rutas de imagen)
        for res in results:
            if not res:
                continue
            fname, boxes = res
            if not fname or not boxes:
                continue
            
            img_path = self._resolve_image(fname)
            if img_path:
                self.samples.append((img_path, boxes))

        print(f"✅ {len(self.samples)} muestras válidas encontradas.")

    def _resolve_image(self, filename: str):
        base = Path(filename)
        for ext in VALID_EXTS:
            cand = self.images_dir / (base.stem + ext)
            if cand.exists():
                return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes_data = self.samples[idx]

        # Cargar imagen con OpenCV para aplicar enhancement
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        # Aplicar enhancement si está habilitado
        if self.use_enhancement:
            img_cv = self.enhancer.enhance(img_cv)
        
        # Convertir de BGR a RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Guardar dimensiones originales
        original_height, original_width = img_cv.shape[:2]
        
        # Convertir a tensor de PyTorch
        img = torch.from_numpy(img_cv).permute(2, 0, 1)  # HWC -> CHW
        img = img[:3]  # Asegurar 3 canales (RGB)

        # --- Redimensionamiento de la imagen ---
        img = F.resize(img, list(self.resize_to), antialias=True)
        img = img.to(torch.float32) / 255.0

        # --- Ajuste de las Bounding Boxes ---
        boxes = torch.tensor([b[1:] for b in boxes_data], dtype=torch.float32)

        # Calcular factores de escala
        x_scale = self.resize_to[1] / original_width
        y_scale = self.resize_to[0] / original_height

        # Aplicar escala a las coordenadas
        boxes[:, 0] *= x_scale  # xmin
        boxes[:, 1] *= y_scale  # ymin
        boxes[:, 2] *= x_scale  # xmax
        boxes[:, 3] *= y_scale  # ymax

        labels = torch.tensor([b[0] for b in boxes_data], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    return model


def plot_history(history, output_dir):
    # (Función de ploteo sin cambios)
    epochs = len(history)
    plt.figure(figsize=(15, 5))
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(
        range(1, epochs + 1), [h["train_loss"] for h in history], label="Train Loss"
    )
    plt.plot(range(1, epochs + 1), [h["val_loss"] for h in history], label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # mAP
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), [h["map"] for h in history], label="mAP")
    plt.plot(range(1, epochs + 1), [h["map_50"] for h in history], label="mAP@.50")
    plt.title("Mean Average Precision")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    # Time
    plt.subplot(1, 3, 3)
    plt.plot(
        range(1, epochs + 1),
        [h["total_time_sec"] for h in history],
        label="Epoch Time",
        color="orange",
    )
    plt.title("Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def evaluate_model(model, dataloader, device):
    metric = MeanAveragePrecision(box_format="xyxy").to(device)
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(
            dataloader, desc="🔍 Validating", unit="batch", leave=False
        ):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            metric.update(predictions, targets)
    stats = metric.compute()
    return stats


def load_checkpoint(checkpoint_path: Path, model, optimizer=None):
    """
    Carga un checkpoint guardado.
    
    Args:
        checkpoint_path: Ruta al archivo .pth
        model: Modelo a cargar
        optimizer: Optimizador (opcional)
        
    Returns:
        Diccionario con información del checkpoint
    """
    print(f"📥 Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Cargar pesos del modelo
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Compatibilidad con modelos antiguos (solo state_dict)
        model.load_state_dict(checkpoint)
    
    # Cargar optimizador si existe
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_map': checkpoint.get('best_map', -1.0),
        'history': checkpoint.get('history', [])
    }
    
    print(f"✅ Checkpoint cargado (Época {info['epoch']}, mAP: {info['best_map']:.4f})")
    return info


def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"🔧 Configuración: Dispositivo={device}, Tamaño Imagen={RESIZE_TO}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * args.train_split)
    train_xml, val_xml = xml_files[:split_idx], xml_files[split_idx:]

    # Crear datasets con o sin enhancement según el argumento
    train_ds = WeaponDetectionDataset(
        Path(args.images_dir), train_xml, resize_to=RESIZE_TO, use_enhancement=args.enhance
    )
    val_ds = WeaponDetectionDataset(
        Path(args.images_dir), val_xml, resize_to=RESIZE_TO, use_enhancement=args.enhance
    )

    num_workers = min(os.cpu_count(), 4) if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = create_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_map = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 1

    # Reanudar desde checkpoint si se especifica
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint_info['epoch'] + 1
            best_map = checkpoint_info['best_map']
            history = checkpoint_info['history']
            print(f"🔄 Reanudando desde época {start_epoch}")
        else:
            print(f"⚠️  Checkpoint no encontrado: {checkpoint_path}")
            print("   Iniciando entrenamiento desde cero...")

    print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
    print(f"📊 Épocas: {start_epoch} → {args.epochs}")
    print(f"💾 Guardando checkpoints cada {args.save_every} épocas")
    print(f"⏸️  Early stopping con paciencia = {args.patience} épocas")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        mem_used_gb = psutil.virtual_memory().used / (1024**3)
        if mem_used_gb > args.ram_limit:
            print(
                f"⛔ Límite de RAM excedido: {mem_used_gb:.2f}GB > {args.ram_limit}GB. Deteniendo."
            )
            break

        print(f"\n--- Epoch {epoch}/{args.epochs} (RAM: {mem_used_gb:.2f}GB) ---")

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc="🏋️ Training", unit="batch")
        for imgs, targets in pbar:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.amp
            ):
                losses_dict = model(imgs, targets)
                losses = sum(loss for loss in losses_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        val_stats = evaluate_model(model, val_loader, device)

        val_loss_accum = 0.0
        model.train()
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_loss_accum += sum(
                    loss for loss in model(imgs, targets).values()
                ).item()
        avg_val_loss = val_loss_accum / len(val_loader)

        epoch_time = time.time() - start_time

        print(
            f"✅ Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mAP: {val_stats['map']:.4f} | Time: {epoch_time:.1f}s"
        )

        epoch_info = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "map": float(val_stats["map"]),
            "map_50": float(val_stats["map_50"]),
            "map_75": float(val_stats["map_75"]),
            "total_time_sec": epoch_time,
        }
        history.append(epoch_info)

        # Guardar mejor modelo por mAP
        if val_stats["map"] > best_map:
            best_map = val_stats["map"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
                'history': history
            }, out_dir / "best_model.pth")
            print(f"💾 ¡Mejor modelo guardado! (mAP: {best_map:.4f})")
        
        # Early stopping basado en validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️  Val Loss no mejoró ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n⏸️  EARLY STOPPING activado! Val Loss no mejoró en {args.patience} épocas")
                print(f"✅ Mejor mAP alcanzado: {best_map:.4f}")
                break
        
        # Guardar checkpoint cada N épocas
        if epoch % args.save_every == 0:
            checkpoint_path = out_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'map': float(val_stats["map"]),
                'best_map': best_map,
                'history': history
            }, checkpoint_path)
            print(f"📦 Checkpoint guardado: {checkpoint_path.name}")

    # Guardar metadata y gráficos al finalizar
    # Guardar metadata y gráficos al finalizar
    (out_dir / "classes.json").write_text(json.dumps({"classes": CLASS_MAP}, indent=2))
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    plot_history(history, str(out_dir))

    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"📊 Épocas completadas: {len(history)}/{args.epochs}")
    print(f"🏆 Mejor mAP alcanzado: {best_map:.4f}")
    print(f"💾 Mejor modelo: {out_dir / 'best_model.pth'}")
    
    # Listar checkpoints guardados
    checkpoints = sorted(out_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        print(f"📦 Checkpoints guardados: {len(checkpoints)}")
        for cp in checkpoints:
            print(f"   - {cp.name}")
    
    print(f"📁 Resultados en: {out_dir}")
    print("=" * 60)


def get_args():
    ap = argparse.ArgumentParser(
        description="Entrenar Faster R-CNN con redimensionamiento para alta velocidad."
    )
    ap.add_argument(
        "--images-dir", default="dataset/images", help="Directorio con imágenes"
    )
    ap.add_argument("--xml-dir", default="dataset/xmls", help="Directorio con XMLs")
    ap.add_argument(
        "--output-dir", default="results_light", help="Directorio de salida"
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Aumentado gracias al redimensionamiento",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-split", type=float, default=0.85)
    ap.add_argument(
        "--device", type=str, default="cuda", help="Dispositivo ('cuda' o 'cpu')"
    )
    ap.add_argument(
        "--amp", action="store_true", help="Activar Automatic Mixed Precision (AMP)"
    )
    ap.add_argument("--ram-limit", type=float, default=28.0, help="Límite de RAM en GB")
    ap.add_argument(
        "--enhance", action="store_true", 
        help="Activar mejoramiento de imágenes (CLAHE, denoising, sharpening)"
    )
    ap.add_argument(
        "--save-every", type=int, default=15,
        help="Guardar checkpoint cada N épocas (default: 15)"
    )
    ap.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping: detener si val_loss no mejora por N épocas (default: 5)"
    )
    ap.add_argument(
        "--resume", type=str, default=None,
        help="Ruta al checkpoint para reanudar entrenamiento (ej: results_light/checkpoint_epoch_30.pth)"
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
