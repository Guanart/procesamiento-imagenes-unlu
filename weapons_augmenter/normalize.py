import cv2
import argparse
from pathlib import Path
import os

DEFAULT_SIZE = (224, 224)

# Normaliza y separa por clase
def normalize_images(input_dir, output_dir, size=DEFAULT_SIZE):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detectar subcarpetas de clase (pistol, knife)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"Se detectaron clases: {[d.name for d in subdirs]}")
        for class_dir in subdirs:
            class_out = output_dir / class_dir.name
            class_out.mkdir(parents=True, exist_ok=True)
            image_files = list(class_dir.glob('*.[jp][pn]g'))
            print(f"Normalizando {len(image_files)} imágenes de '{class_dir.name}'...")
            for i, image_path in enumerate(image_files):
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Error al leer {image_path.name}, se omite.")
                    continue
                norm_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                out_path = class_out / image_path.name
                cv2.imwrite(str(out_path), norm_img)
                print(f"[{i+1}/{len(image_files)}] {image_path.name} -> {out_path}")
    print(f"\n--- Proceso de normalización completado ---\nDirectorio de salida: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normaliza imágenes a tamaño fijo y separa por clase.")
    parser.add_argument('--input', type=str, required=True, help='Directorio con imágenes a normalizar (ej: dataset/augmented)')
    parser.add_argument('--output', type=str, required=True, help='Directorio de salida para imágenes normalizadas (ej: dataset/normalized)')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Tamaño destino (ancho alto), default 224x224')
    args = parser.parse_args()
    normalize_images(args.input, args.output, tuple(args.size))
