import os
from PIL import Image
import torch
from torchvision import transforms
import json
import argparse
import numpy as np

DEFAULT_OUTPUT_DIR = "results"
APPLY_NORMALIZATION = False
APPLY_RESIZE = False

def add_args(parser):
    parser.add_argument("--amp", action="store_true", help="Usar mixed precision en inferencia")
    parser.add_argument("--channels-last", action="store_true", help="Usar memory_format channels_last")

# Cargar clases desde experiment_meta.json
def load_class_names(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["class_names"]

def build_transform():
    ops = []
    if APPLY_RESIZE:
        ops.append(transforms.Resize((224, 224)))
    ops.append(transforms.ToTensor())
    if APPLY_NORMALIZATION:
        ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def predict_image(model, image_path, class_names, device, channels_last: bool, amp: bool):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    # Construcción explícita del tensor evita problemas de tipado estático
    np_img = np.array(image)  # shape [H,W,C]
    img_tensor = torch.from_numpy(np_img).permute(2,0,1).to(torch.float32) / 255.0
    tensor = img_tensor.unsqueeze(0).to(device)
    if channels_last:
        tensor = tensor.to(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', dtype=torch.float16, enabled=amp), torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    print(f"Predicción: {class_names[pred.item()]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de modelo de clasificación de armas optimizada")
    parser.add_argument("--model_path", type=str, default=f"{DEFAULT_OUTPUT_DIR}/best_model.pth", help="Ruta al modelo entrenado (.pth)")
    parser.add_argument("--meta_path", type=str, default=f"{DEFAULT_OUTPUT_DIR}/experiment_meta.json", help="Ruta al archivo de metadatos")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen a clasificar")
    parser.add_argument("--device", type=str, default="cuda", help="cpu o cuda")
    add_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.meta_path)

    # Cargar modelo
    import torchvision.models as models
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # channels_last se aplica a los tensores de entrada, no al módulo directamente

    # Inferencia con autocast si se solicita
    model.eval()
    predict_image(model, args.image, class_names, device, channels_last=args.channels_last, amp=args.amp)