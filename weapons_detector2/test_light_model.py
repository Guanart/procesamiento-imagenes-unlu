import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from pathlib import Path

# --- CONFIGURACIÓN POR DEFECTO ---
DEFAULT_MODEL_PATH = "results_light/best_model.pth"
DEFAULT_IMAGE_PATH = "dataset/test/test_image.jpg"
DEFAULT_OUTPUT_DIR = "test_results"
DEFAULT_CONFIDENCE = 0.5

# Clases (deben coincidir con las usadas en el entrenamiento)
# El orden es importante: __background__ va primero.
CLASSES = ["__background__", "knife", "pistol"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


def test_single_image(model_path, image_path, output_dir, confidence_threshold):
    """Carga el modelo entrenado, realiza inferencia en una imagen y guarda el resultado."""
    print(f"Usando dispositivo: {DEVICE}")

    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en '{model_path}'.")
        print("Asegúrate de que el entrenamiento haya finalizado y el archivo exista.")
        return

    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen de prueba en '{image_path}'.")
        return

    print(f"Cargando modelo desde '{model_path}'...")
    model = get_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Modelo cargado y en modo de evaluación.")

    print(f"Cargando imagen: '{image_path}'")
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():
        print("Realizando inferencia...")
        predictions = model([image_tensor])

    # Procesar y dibujar resultados
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        print("Fuente 'arial.ttf' no encontrada. Usando fuente por defecto.")

    detections_found = 0
    for box, label, score in zip(
        predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]
    ):
        if score >= confidence_threshold:
            detections_found += 1
            class_name = CLASSES[label.item()]

            draw.rectangle(box.tolist(), outline="red", width=3)
            text = f"{class_name}: {score:.2f}"

            # Lógica para dibujar texto con fondo
            try:
                bbox = draw.textbbox((box[0], box[1]), text, font=font)
                draw.rectangle(bbox, fill="red")
                draw.text((box[0], box[1]), text, fill="white", font=font)
            except AttributeError:  # Fallback para versiones antiguas de Pillow
                draw.text((box[0], box[1] - 10), text, fill="red", font=font)

    print(
        f"Se encontraron {detections_found} detecciones con una confianza >= {confidence_threshold}."
    )

    # Generar nombre de salida basado en el nombre de la imagen de entrada
    input_path = Path(image_path)
    output_filename = f"{input_path.stem}_detected{input_path.suffix}"
    
    # Guardar imagen resultante
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    print(f"Imagen con detecciones guardada en: '{output_path}'")


def get_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Prueba el modelo Faster R-CNN entrenado en una imagen."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al modelo entrenado (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help=f"Ruta a la imagen de prueba (default: {DEFAULT_IMAGE_PATH})",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_single_image(args.model, args.image, args.output, args.confidence)
