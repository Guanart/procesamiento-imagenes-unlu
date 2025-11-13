import os
from PIL import Image
import torch
from torchvision import transforms
import json
import argparse

DEFAULT_OUTPUT_DIR = "results"
APPLY_NORMALIZATION = False
APPLY_RESIZE = False

# Cargar clases desde experiment_meta.json
def load_class_names(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["class_names"]

def predict_image(model, image_path, class_names, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform_list = [transforms.ToTensor()]
    if APPLY_RESIZE:
        transform_list.insert(0, transforms.Resize((224, 224)))
    if APPLY_NORMALIZATION:
        transform_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    basic_transform = transforms.Compose(transform_list)
    input_tensor = basic_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    print(f"Predicción: {class_names[pred.item()]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de modelo de clasificación de armas")
    parser.add_argument("--model_path", type=str, default=f"{DEFAULT_OUTPUT_DIR}/best_model.pth", help="Ruta al modelo entrenado (.pth)")
    parser.add_argument("--meta_path", type=str, default=f"{DEFAULT_OUTPUT_DIR}/experiment_meta.json", help="Ruta al archivo de metadatos")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen a clasificar")
    parser.add_argument("--device", type=str, default="cuda", help="cpu o cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.meta_path)

    # Cargar modelo
    import torchvision.models as models
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    predict_image(model, args.image, class_names, device)