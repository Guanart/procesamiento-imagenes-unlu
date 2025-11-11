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


def get_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento clasificación armas (dataset ya saneado)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio raíz con subcarpetas train/val/pistol/knife",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def train_model(
    model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, output_dir
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_model_wts = model.state_dict()

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

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double().item() / total

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()

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
        f.write(report)

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


def main():
    args = get_args()
    make_dirs(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Asumimos que las imágenes ya tienen tamaño correcto — sólo convertimos a tensor + normalizamos mínimamente:
    from torchvision import transforms

    basic_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_dir, x), basic_transform)
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=(x == "train"),
            num_workers=4,
        )
        for x in ["train", "val"]
    }
    class_names = image_datasets["train"].classes

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
