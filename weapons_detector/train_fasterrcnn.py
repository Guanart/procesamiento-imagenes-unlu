#!/usr/bin/env python3
"""
Entrenamiento Faster R-CNN para detección de armas (knife, pistol) usando anotaciones Pascal VOC (.xml).

Requisitos:
  pip install torch torchvision
  (GPU opcional pero recomendado)

Uso ejemplo:
  python weapons_detector/train_fasterrcnn.py \
      --images-dir path/JPEGImages \
      --xml-dir path/Annotations \
      --output-dir results_frcnn \
      --epochs 15 --batch-size 4

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

# Configuración ROCm para AMD (reduce fragmentación de memoria)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET

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
    def __init__(self, images_dir: Path, xml_files: List[Path], transforms=None):
        self.images_dir = images_dir
        self.xml_files = xml_files
        self.transforms = transforms
        self.samples = []  # list of (image_path, boxes)
        for xml in xml_files:
            fname, boxes = parse_voc_xml(xml)
            if not fname:
                continue
            img_path = self._resolve_image(fname)
            if img_path is None:
                continue
            self.samples.append((img_path, boxes))

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
        img_path, boxes = self.samples[idx]
        # Cargar imagen
        img = torchvision.io.read_image(str(img_path)).to(torch.float32) / 255.0  # [C,H,W]
        img = img[:3]  # asegurar 3 canales
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
    # Aplicar solo flip horizontal simple (imagen ya normalizada 0-1)
    def _tf(img):
        if train and random.random() < 0.5:
            img = F.hflip(img)
        return img
    return _tf


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate_epoch(model, dataloader, device) -> Dict:
    model.eval()
    stats = {"images":0, "detections":0, "gt":0, "mean_iou":0.0}
    iou_accum = []
    with torch.no_grad():
        for imgs, targets in dataloader:
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

    random.shuffle(xml_files)
    split = int(len(xml_files) * args.train_split)
    train_xml = xml_files[:split]
    val_xml = xml_files[split:]

    train_ds = WeaponDetectionDataset(images_dir, train_xml, transforms=get_transforms(train=True))
    val_ds   = WeaponDetectionDataset(images_dir, val_xml, transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(NUM_CLASSES).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for imgs, targets in train_loader:
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
        lr_scheduler.step()
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        val_stats = evaluate_epoch(model, val_loader, device)
        # Obtener pérdida de validación "proxy" usando forward con targets
        val_loss_accum = 0.0
        model.train()  # para permitir cálculo de pérdida
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                losses_dict = model(imgs, targets)
                val_loss_accum += sum(loss for loss in losses_dict.values()).item()
        avg_val_loss = val_loss_accum / max(1, len(val_loader))

        epoch_info = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mean_iou': val_stats['mean_iou'],
            'val_images': val_stats['images'],
            'val_detections': val_stats['detections'],
            'val_gt': val_stats['gt'],
            'time_sec': time.time() - start
        }
        history.append(epoch_info)
        print(f"Epoch {epoch}/{args.epochs} - train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} meanIoU={val_stats['mean_iou']:.3f}")

        # Guardar si mejora
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            print("  -> Modelo mejorado guardado")

    # Guardar historia y clases
    (out_dir / 'classes.json').write_text(json.dumps({'classes': CLASS_MAP}, indent=2))
    (out_dir / 'training_log.json').write_text(json.dumps(history, indent=2))
    print("\nEntrenamiento completado. Modelo guardado en:", out_dir / 'best_model.pth')


def get_args():
    ap = argparse.ArgumentParser(description='Entrenar Faster R-CNN knife/pistol (optimizado ROCm/AMD)')
    ap.add_argument('--images-dir', required=True, help='Directorio con imágenes (ej: images/)')
    ap.add_argument('--xml-dir', required=True, help='Directorio con XML (ej: xmls/)')
    ap.add_argument('--output-dir', default='results_frcnn', help='Salida modelos y logs')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--train-split', type=float, default=0.8)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--amp', action='store_true', help='Activar mixed precision')
    return ap.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
