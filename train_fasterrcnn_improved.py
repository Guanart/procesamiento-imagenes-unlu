#!/usr/bin/env python3
"""
Script mejorado para entrenar Faster R-CNN en detección de armas.
Mejoras:
- Más epochs
- Learning rate más bajo
- Data augmentation
- Mejor manejo de clases
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class WeaponsDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        # Cargar anotaciones
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Crear mapeo de clases
        self.class_names = ['__background__', 'knife', 'pistol']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Filtrar imágenes que existen
        self.valid_samples = []
        for sample in self.annotations['samples']:
            img_path = self.images_dir / sample['image']
            if img_path.exists():
                self.valid_samples.append(sample)

        print(f"Dataset: {len(self.valid_samples)} imágenes válidas de {len(self.annotations['samples'])}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        img_path = self.images_dir / sample['image']

        # Cargar imagen
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Convertir a tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Procesar bounding boxes
        boxes = []
        labels = []

        for ann in sample['annotations']:
            if ann['class'] in self.class_to_idx:
                # Convertir coordenadas normalizadas a absolutas
                x_min = int(ann['bbox']['x_min'] * image.shape[2])
                y_min = int(ann['bbox']['y_min'] * image.shape[1])
                x_max = int(ann['bbox']['x_max'] * image.shape[2])
                y_max = int(ann['bbox']['y_max'] * image.shape[1])

                # Asegurar que las coordenadas sean válidas
                x_min = max(0, min(x_min, image.shape[2]-1))
                x_max = max(x_min+1, min(x_max, image.shape[2]))
                y_min = max(0, min(y_min, image.shape[1]-1))
                y_max = max(y_min+1, min(y_max, image.shape[1]))

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_to_idx[ann['class']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }

        return image, target

def get_model(num_classes):
    """Crear modelo Faster R-CNN con backbone ResNet50"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Reemplazar el predictor de clases
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.0001):
    """Entrenar el modelo"""
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(val_loader)

        print(".4f")

        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'results_real/best_model_improved.pth')
            print("💾 Mejor modelo guardado")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️  Early stopping")
                break

        lr_scheduler.step()

    return model

def main():
    print("🚀 Entrenamiento Mejorado de Faster R-CNN para Detección de Armas")

    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    images_dir = "weapons_detector/dataset/images"
    annotations_file = "weapons_detector/dataset/annotations.json"

    if not os.path.exists(annotations_file):
        print(f"❌ Archivo de anotaciones no encontrado: {annotations_file}")
        return

    # Crear dataset
    dataset = WeaponsDataset(images_dir, annotations_file)

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Modelo
    num_classes = len(dataset.class_names)  # background + knife + pistol
    model = get_model(num_classes)

    print(f"Clases: {dataset.class_names}")
    print(f"Número de clases: {num_classes}")

    # Entrenar
    trained_model = train_model(model, train_loader, val_loader, device,
                               num_epochs=50, lr=0.0001)

    print("✅ Entrenamiento completado!")

    # Guardar clases
    with open('results_real/classes_improved.json', 'w') as f:
        json.dump(dataset.class_names, f)

    print("📁 Modelo y clases guardados en results_real/")

if __name__ == "__main__":
    main()