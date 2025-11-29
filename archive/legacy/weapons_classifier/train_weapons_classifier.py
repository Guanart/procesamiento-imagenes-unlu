# train_weapon_classifier_simple.py

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torchvision import transforms
import random
from PIL import Image

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64  # Ajustable según memoria
DEFAULT_LR = 0.001
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_WEIGHT_DECAY = 0.01

TRAIN_SPLIT = 0.8  # Proporción de train/val


def get_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento clasificación armas optimizado (ROCm / CUDA / CPU)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directorio con subcarpetas por clase (pistol/knife).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / mps")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "cosine", "plateau", "none"], help="Tipo scheduler")
    parser.add_argument("--amp", action="store_true", help="Activar mixed precision (torch.autocast)")
    parser.add_argument("--channels-last", action="store_true", help="Usar memory_format=channels_last para acelerar convoluciones")
    parser.add_argument("--freeze-backbone", action="store_true", help="Congelar capas iniciales y entrenar solo la FC")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing para CrossEntropy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    num_epochs,
    output_dir,
    use_amp: bool,
    channels_last: bool,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_model_wts = model.state_dict()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0
            num_batches = len(dataloaders[phase])
            batch_count = 0

            start_iter = time.time()
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                batch_count += 1
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=torch.float16, enabled=use_amp):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)

                # Medidor de progreso por batch
                iter_time = time.time() - start_iter
                imgs_per_sec = inputs.size(0) / iter_time if iter_time > 0 else 0
                print(
                    f"  [{phase}] Batch {batch_count}/{num_batches} - {total} imgs acumuladas | {imgs_per_sec:.1f} img/s",
                    end="\r",
                )

            print()  # Salto de línea tras el último batch
            epoch_loss = running_loss / total
            # running_corrects puede ser tensor; convertir seguro
            running_corrects_tensor = torch.as_tensor(running_corrects)
            epoch_acc = (float(running_corrects_tensor.item()) / total) if total > 0 else 0.0

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # Para ReduceLROnPlateau, pasar el loss de validación
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history["val_loss"][-1])
            else:
                scheduler.step()

        # Print del learning rate actual
        for param_group in optimizer.param_groups:
            print(f"Learning rate actual: {param_group['lr']}")

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    return model, history


def evaluate_model(model, dataloader, device, class_names, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    np.savetxt(
        os.path.join(output_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d"
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(str(report))

    return cm, report


def plot_history(history, output_dir):
    epochs = len(history["train_loss"])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="train_loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="train_acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main():
    args = get_args()
    make_dirs(args.output_dir)
    set_seed(args.seed)

    # Selección dispositivo (ROCm usa 'cuda')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA/ROCm no disponible, usando CPU.")
        device = torch.device('cpu')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Las imágenes ya fueron aumentadas (simple_augmenter.py) y normalizadas (normalize.py)
    # Solo convertir a tensor sin transformaciones adicionales
    basic_transform = transforms.ToTensor()

    # Leer todas las imágenes y etiquetas
    all_images = []
    all_labels = []
    class_names = []
    for class_name in sorted(os.listdir(args.data_dir)):
        class_path = os.path.join(args.data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                all_images.append(os.path.join(class_path, fname))
                all_labels.append(class_names.index(class_name))

    # Mezclar y hacer split
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * TRAIN_SPLIT)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    # Dataset personalizado
    class SimpleImageDataset(torch.utils.data.Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            img_path, label = self.items[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label

    image_datasets = {
        "train": SimpleImageDataset(train_data, basic_transform),
        "val": SimpleImageDataset(val_data, basic_transform),
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=(x == "train"),
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if torch.cuda.is_available() else False,
        )
        for x in ["train", "val"]
    }

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
        print("Backbone congelado, solo entrenando la capa fully-connected.")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(dataloaders['train']),
            epochs=args.epochs,
            pct_start=0.1,
            anneal_strategy='cos',
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    else:
        scheduler = None

    start = time.time()
    model, history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        device,
        args.epochs,
        args.output_dir,
        use_amp=args.amp,
        channels_last=args.channels_last,
    )
    train_time = time.time() - start
    print(f"Training complete in {train_time // 60:.0f}m {train_time % 60:.0f}s")

    plot_history(history, args.output_dir)
    cm, report = evaluate_model(
        model, dataloaders["val"], device, class_names, args.output_dir
    )

    meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "channels_last": args.channels_last,
        "train_time_seconds": train_time,
        "class_names": class_names,
        "history_last": {
            "train_acc": history["train_acc"][-1],
            "val_acc": history["val_acc"][-1],
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
        },
    }
    with open(os.path.join(args.output_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
