#!/usr/bin/env python3
"""
Entrenamiento Faster R-CNN Ligero (MobileNetV3) para detección de armas (knife, pistol) usando anotaciones Pascal VOC (.xml).

Esta versión optimizada usa MobileNetV3 como backbone para entrenamiento eficiente en CPU,
logrando ~30-60x speedup comparado con ResNet50 mientras mantiene precisión competitiva.

Requisitos:
  pip install torch torchvision
  (GPU opcional pero recomendado)

Uso ejemplo:
  python train_fasterrcnn_light.py \
      --images-dir path/JPEGImages \
      --xml-dir path/Annotations \
      --output-dir results_light \
      --epochs 15 --batch-size 2

Salida:
  - best_model.pth (pesos del modelo)
  - classes.json (mapa de clases)
  - training_log.json (historial de pérdidas)

Estructura targets por imagen (torchvision):
  target = {
    'boxes': Tensor[N,4] (xmin,ymin,xmax,ymax)
    'labels': Tensor[N] (>=1, 0 es background implícito)
    'image_id': Tensor[1]
    'area': Tensor[N]
    'iscrowd': Tensor[N] (0)
  }
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

# Configuración ROCm para AMD (reduce fragmentación de memoria)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET
from tqdm import tqdm
import psutil

# Mapa de clases (background implícito): labels deben iniciar en 1
CLASS_MAP = {"knife": 1, "pistol": 2}
NUM_CLASSES = len(CLASS_MAP) + 1  # + background
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_voc_xml(xml_path: Path) -> Tuple[str, List[Tuple[int,int,int,int,int]]]:
    """
    Devuelve (filename, lista de cajas: (label,xmin,ymin,xmax,ymax))
    Ignora <path> (puede estar mal), solo usa <filename>
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError:
        return "", []
    
    # Extraer filename (ignorar <path>)
    filename_el = root.find('filename')
    if filename_el is None or not filename_el.text:
        # Fallback: usar nombre del XML
        filename = xml_path.stem + '.jpg'
    else:
        filename = filename_el.text.strip()
        # Si no tiene extensión, agregar .jpg por defecto
        if '.' not in filename:
            filename = filename + '.jpg'
    
    boxes = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        if name_el is None or not name_el.text:
            continue
        cls_name = name_el.text.strip().lower()
        if cls_name not in CLASS_MAP:
            continue
        bbox_el = obj.find('bndbox')
        if bbox_el is None:
            continue
        try:
            xmin_el = bbox_el.find('xmin')
            ymin_el = bbox_el.find('ymin')
            xmax_el = bbox_el.find('xmax')
            ymax_el = bbox_el.find('ymax')
            if None in (xmin_el, ymin_el, xmax_el, ymax_el):
                continue
            xmin = int(float(xmin_el.text or "0"))
            ymin = int(float(ymin_el.text or "0"))
            xmax = int(float(xmax_el.text or "0"))
            ymax = int(float(ymax_el.text or "0"))
        except (ValueError, AttributeError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((CLASS_MAP[cls_name], xmin, ymin, xmax, ymax))
    return filename, boxes


class WeaponDetectionDataset(Dataset):
    def __init__(self, images_dir: Path, xml_files: List[Path], transforms=None, max_images_per_class=None):
        self.images_dir = images_dir
        self.xml_files = xml_files
        self.transforms = transforms
        self.max_images_per_class = max_images_per_class
        self.samples = []  # list of (image_path, boxes)
        
        print(f"📦 Parseando {len(xml_files)} archivos XML...")
        for xml in tqdm(xml_files, desc="Parseando XMLs", leave=False):
            fname, boxes = parse_voc_xml(xml)
            if not fname:
                continue
            img_path = self._resolve_image(fname)
            if img_path is None:
                continue
            self.samples.append((img_path, boxes))
        
        # Filtrar por máximo imágenes por clase
        if self.max_images_per_class is not None:
            count_per_class = {cls: 0 for cls in CLASS_MAP.values()}
            filtered_samples = []
            for img_path, boxes in self.samples:
                classes_in_img = set(b[0] for b in boxes)
                can_add = all(count_per_class[cls] < self.max_images_per_class for cls in classes_in_img)
                if can_add:
                    filtered_samples.append((img_path, boxes))
                    for cls in classes_in_img:
                        count_per_class[cls] += 1
            self.samples = filtered_samples
        
        print(f"✅ {len(self.samples)} muestras cargadas (filtradas por clase)")
        
        # Cachear imágenes en memoria
        self.cached_images = []
        print("📦 Cacheando imágenes en memoria...")
        for img_path, _ in tqdm(self.samples, desc="Cacheando imágenes", leave=False):
            img = torchvision.io.read_image(str(img_path)).to(torch.float32) / 255.0
            img = img[:3]  # asegurar 3 canales
            self.cached_images.append(img)
        print(f"✅ {len(self.cached_images)} imágenes cacheadas")

    def _resolve_image(self, filename: str):
        base = Path(filename)
        if base.suffix.lower() in VALID_EXTS:
            cand = self.images_dir / base.name
            if cand.exists():
                return cand
        else:
            for ext in VALID_EXTS:
                cand = self.images_dir / (base.stem + ext)
                if cand.exists():
                    return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        _, boxes = self.samples[idx]
        # Construir target
        if boxes:
            box_tensor = torch.tensor([b[1:] for b in boxes], dtype=torch.float32)
            labels = torch.tensor([b[0] for b in boxes], dtype=torch.int64)
        else:
            box_tensor = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        area = (box_tensor[:,2] - box_tensor[:,0]) * (box_tensor[:,3] - box_tensor[:,1]) if len(box_tensor)>0 else torch.zeros((0,), dtype=torch.float32)
        target = {
            'boxes': box_tensor,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target


def get_transforms(train: bool=True):
    # No transformations applied
    return None


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def plot_history(history, output_dir):
    epochs = len(history["train_loss"])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="train_loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), [h["val_mean_iou"] for h in history], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.title("Validation IoU")
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs + 1), [h["val_detections"] for h in history], label="detections")
    plt.plot(range(1, epochs + 1), [h["val_gt"] for h in history], label="ground_truth")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Detections vs Ground Truth")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    model.eval()
    stats = {"images":0, "detections":0, "gt":0, "mean_iou":0.0}
    iou_accum = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="🔍 Validating", unit="batch") if show_progress else dataloader
        for imgs, targets in iterator:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                stats["images"] += 1
                stats["gt"] += len(tgt['boxes'])
                stats["detections"] += len(out['boxes'])
                # Calcular IoU promedio simple emparejando cajas por máxima IoU (heurística)
                if len(tgt['boxes']) and len(out['boxes']):
                    ious = box_iou(out['boxes'].cpu(), tgt['boxes'])  # [D, G]
                    # Para cada gt tomar mejor detection
                    best = ious.max(dim=0).values
                    iou_accum.extend(best.tolist())
    if iou_accum:
        stats['mean_iou'] = sum(iou_accum)/len(iou_accum)
    return stats


def plot_history(history, output_dir):
    epochs = len(history)
    plt.figure(figsize=(12, 8))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), [h['train_loss'] for h in history], label="train_loss")
    plt.plot(range(1, epochs + 1), [h['val_loss'] for h in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    # IoU
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), [h['val_mean_iou'] for h in history], label="val_mean_iou", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Validation Mean IoU")
    plt.legend()
    
    # Detections
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs + 1), [h['val_detections'] for h in history], label="detections", color='red')
    plt.plot(range(1, epochs + 1), [h['val_gt'] for h in history], label="ground_truth", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.title("Detections vs Ground Truth")
    plt.legend()
    
    # Time
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs + 1), [h['total_time_sec'] for h in history], label="epoch_time", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Epoch Training Time")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def evaluate_model(model, dataloader, device, output_dir):
    model.eval()
    total_iou = 0.0
    num_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions (sin targets para inferencia)
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                num_samples += 1
                if len(out['boxes']) > 0 and len(tgt['boxes']) > 0:
                    ious = box_iou(out['boxes'].cpu(), tgt['boxes'])
                    max_iou = ious.max().item() if ious.numel() > 0 else 0.0
                    total_iou += max_iou
                    
                    # Para confusión matrix
                    matched = (ious > 0.5).any(dim=0)
                    for i, label in enumerate(tgt['labels']):
                        pred_label = out['labels'][ious[:, i].argmax()].item() if matched[i] else 0
                        all_preds.append(pred_label)
                        all_labels.append(label.item())
                else:
                    total_iou += 0.0
                    for label in tgt['labels']:
                        all_preds.append(0)
                        all_labels.append(label.item())
    
    mean_iou = total_iou / num_samples if num_samples > 0 else 0.0
    
    # Matriz de confusión
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(all_preds, all_labels):
        if p < NUM_CLASSES and l < NUM_CLASSES:
            cm[l, p] += 1
    
    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    
    report = f"Total predictions: {len(all_preds)}\nTotal labels: {len(all_labels)}\nConfusion Matrix:\n{cm}"
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
        f.write(report)
    
    print("Evaluation Report:")
    print(report)
    return {
        'val_loss': 0.0, 
        'mean_iou': mean_iou,
        'images': num_samples,
        'detections': len(all_preds),
        'gt': len(all_labels)
    }


def train(args):
    xml_dir = Path(args.xml_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verificación GPU/ROCm
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"🔧 Configuración de Entrenamiento")
    print("=" * 60)
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Memoria GPU: {mem_gb:.2f} GB")
    print(f"Mixed Precision (AMP): {args.amp}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    xml_files = sorted(xml_dir.glob('*.xml'))
    if not xml_files:
        raise RuntimeError(f"No se encontraron XML en {xml_dir}")

    print(f"📂 Total de archivos XML encontrados: {len(xml_files)}")
    
    random.shuffle(xml_files)
    split = int(len(xml_files) * args.train_split)
    train_xml = xml_files[:split]
    val_xml = xml_files[split:]
    
    print(f"📊 Split dataset: {len(train_xml)} train, {len(val_xml)} val")
    print(f"\n🔄 Cargando datasets...")

    # Crear datasets sin cachear en RAM (evita crashes por memoria)
    train_ds = WeaponDetectionDataset(images_dir, train_xml, transforms=None, max_images_per_class=args.max_images_per_class)
    val_ds   = WeaponDetectionDataset(images_dir, val_xml, transforms=None, max_images_per_class=args.max_images_per_class)
    
    print(f"✅ Train dataset: {len(train_ds)} imágenes")
    print(f"✅ Val dataset: {len(val_ds)} imágenes")

    # Optimizar DataLoader para CPU: num_workers=0, sin pin_memory
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"\n🔢 Batches por época:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🤖 Creando modelo Faster R-CNN...")
    model = create_model(NUM_CLASSES).to(device)
    print(f"✅ Modelo cargado en {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_loss = float('inf')
    history = []
    
    print(f"\n" + "="*60)
    print(f"🚀 INICIANDO ENTRENAMIENTO")
    print("="*60)

    for epoch in range(1, args.epochs+1):
        # Chequear límite de RAM
        mem_used_gb = psutil.virtual_memory().used / (1024**3)
        if mem_used_gb > args.ram_limit:
            print(f"RAM limit exceeded: {mem_used_gb:.2f}GB > {args.ram_limit}GB, stopping training")
            break
        
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # FASE DE ENTRENAMIENTO
        model.train()
        epoch_loss = 0.0
        start = time.time()
        
        # Usar tqdm para barra de progreso
        batch_count = 0
        train_pbar = tqdm(train_loader, desc="🏋️  Training", unit="batch", ncols=100)
        for imgs, targets in train_pbar:
            batch_count += 1
            if batch_count == 1:
                print(f"\n⏳ Procesando primer batch (puede tardar ~30-60s)...", flush=True)
            
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                losses_dict = model(imgs, targets)
                losses = sum(loss for loss in losses_dict.values())
            
            if args.amp:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            
            epoch_loss += losses.item()
            
            # Actualizar barra de progreso con loss actual
            train_pbar.set_postfix({'loss': f'{losses.item():.4f}'})
        lr_scheduler.step()
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_time = time.time() - start
        print(f"✅ [TRAIN] Loss: {avg_train_loss:.4f} | Tiempo: {train_time:.1f}s")
        
        # FASE DE VALIDACIÓN
        print(f"\n🔍 [VAL] Evaluando...")
        val_start = time.time()
        val_stats = evaluate_model(model, val_loader, device, args.output_dir)
        avg_val_loss = val_stats['val_loss']
        
        # Obtener pérdida de validación "proxy" usando forward con targets
        val_loss_accum = 0.0
        model.train()  # para permitir cálculo de pérdida
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="📊 Val Loss", unit="batch")
            for imgs, targets in val_pbar:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                losses_dict = model(imgs, targets)
                batch_loss = sum(loss for loss in losses_dict.values()).item()
                val_loss_accum += batch_loss
                val_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        avg_val_loss = val_loss_accum / max(1, len(val_loader))
        val_time = time.time() - val_start
        print(f"✅ [VAL] IoU: {val_stats['mean_iou']:.3f} | Tiempo: {val_time:.1f}s")

        epoch_info = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mean_iou': val_stats['mean_iou'],
            'val_images': val_stats['images'],
            'val_detections': val_stats['detections'],
            'val_gt': val_stats['gt'],
            'train_time_sec': train_time,
            'val_time_sec': val_time,
            'total_time_sec': time.time() - start
        }
        history.append(epoch_info)
        
        # Mostrar learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Learning rate actual: {current_lr:.6f}")
        
        # Estadísticas de detección
        print(f"📈 Detecciones: {val_stats['detections']}/{val_stats['gt']} (pred/gt) en {val_stats['images']} imágenes")

        # Guardar si mejora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            print(f"💾 ¡Mejor modelo guardado! (val_loss: {best_val_loss:.4f})")
        
        # Tiempo total de época
        epoch_total_time = time.time() - start
        print(f"⏱️  Tiempo total época: {epoch_total_time:.1f}s ({epoch_total_time/60:.1f}m)")

    # Guardar historia y clases
    (out_dir / 'classes.json').write_text(json.dumps({'classes': CLASS_MAP}, indent=2))
    (out_dir / 'training_log.json').write_text(json.dumps(history, indent=2))
    
    # Generar gráficos y métricas finales
    plot_history(history, str(out_dir))
    final_stats = evaluate_model(model, val_loader, device, str(out_dir))
    
    # Metadata
    meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_split": args.train_split,
        "device": str(device),
        "amp": args.amp,
        "ram_limit": args.ram_limit,
        "max_images_per_class": args.max_images_per_class,
        "total_train_time_sec": sum(h['total_time_sec'] for h in history),
        "final_train_loss": history[-1]['train_loss'],
        "final_val_loss": history[-1]['val_loss'],
        "final_val_iou": history[-1]['val_mean_iou'],
        "best_val_loss": min(h['val_loss'] for h in history),
        "classes": CLASS_MAP,
        "num_classes": NUM_CLASSES
    }
    with open(os.path.join(out_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\nEntrenamiento completado. Modelo guardado en:", out_dir / 'best_model.pth')
    print("Gráficos y métricas generadas en:", out_dir)


def get_args():
    ap = argparse.ArgumentParser(description='Entrenar Faster R-CNN knife/pistol (optimizado ROCm/AMD)')
    ap.add_argument('--images-dir', required=True, help='Directorio con imágenes (ej: images/)')
    ap.add_argument('--xml-dir', required=True, help='Directorio con XML (ej: xmls/)')
    ap.add_argument('--output-dir', default='results_light', help='Salida modelos y logs')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=2, help='Batch size (reducido para CPU)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--train-split', type=float, default=0.8)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--amp', action='store_true', help='Activar mixed precision')
    ap.add_argument('--ram-limit', type=float, default=20.0, help='Límite de RAM en GB para detener entrenamiento')
    ap.add_argument('--max-images-per-class', type=int, default=500, help='Máximo de imágenes por clase (None para ilimitado)')
    return ap.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
