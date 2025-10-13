import cv2
import argparse
from pathlib import Path
import shutil

class SimpleAugmenter:
    """
    Una clase simple para aumentar un dataset de imágenes aplicando transformaciones básicas.

    Atributos:
        input_dir (str): Directorio de las imágenes originales.
        output_dir (str): Directorio donde se guardarán las imágenes aumentadas.
    """

    def __init__(self, input_dir, output_dir):
        """
        Inicializa el aumentador con los directorios de entrada y salida.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"El directorio de entrada no existe: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_transformations(self, image_path):
        """
        Aplica un conjunto fijo de transformaciones a una sola imagen.

        Args:
            image_path (Path): La ruta a la imagen a transformar.

        Returns:
            dict: Un diccionario con los nombres de las transformaciones y las imágenes (numpy arrays).
        """
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            return {}

        # 1. Volteo Horizontal
        h_flip = cv2.flip(original_image, 1)

        # 2. Volteo Vertical
        v_flip = cv2.flip(original_image, 0)

        # 3. Rotación 90 grados
        rotated_90 = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

        return {
            "h_flip": h_flip,
            "v_flip": v_flip,
            "rotated_90": rotated_90,
        }

    def process(self):
        """
        Procesa todas las imágenes en el directorio de entrada, aplica las transformaciones
        y guarda los resultados en el directorio de salida.
        """
        print(f"Iniciando aumentación simple desde '{self.input_dir}' hacia '{self.output_dir}'...")
        image_files = list(self.input_dir.glob('*.[jp][pn]g'))
        
        if not image_files:
            print("Advertencia: No se encontraron imágenes .jpg o .png en el directorio de entrada.")
            return

        total_original = len(image_files)
        total_augmented = 0

        for i, image_path in enumerate(image_files):
            print(f"Procesando [{i+1}/{total_original}]: {image_path.name}")
            
            # Copia la imagen original
            original_dest = self.output_dir / image_path.name
            cv2.imwrite(str(original_dest), cv2.imread(str(image_path)))
            total_augmented += 1

            transformations = self._apply_transformations(image_path)

            for transform_name, transformed_image in transformations.items():
                new_filename = f"{image_path.stem}_{transform_name}{image_path.suffix}"
                output_path = self.output_dir / new_filename
                cv2.imwrite(str(output_path), transformed_image)
                total_augmented += 1
        
        print("\n--- Proceso de Aumentación Completado ---")
        print(f"Imágenes originales procesadas: {total_original}")
        print(f"Imágenes totales en el directorio de salida: {total_augmented}")
        print(f"Directorio de salida: '{self.output_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Aumentador simple de imágenes para datasets de armas.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default='dataset/original',
        help='Directorio con las imágenes originales (por clase, ej: dataset/original/knife).'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset/augmented',
        help='Directorio para guardar las imágenes aumentadas (por clase, ej: dataset/augmented/knife).'
    )
    
    args = parser.parse_args()

    # Detectar si hay subdirectorios de clases (knife, pistol)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if subdirs:
        print(f"Se detectaron {len(subdirs)} clases: {[d.name for d in subdirs]}")
        for class_dir in subdirs:
            print(f"\n--- Procesando clase: {class_dir.name} ---")
            class_output_dir = output_path / class_dir.name
            augmenter = SimpleAugmenter(class_dir, class_output_dir)
            augmenter.process()
    else:
        print("\n--- Procesando en modo de directorio único ---")
        augmenter = SimpleAugmenter(args.input, args.output)
        augmenter.process()

if __name__ == '__main__':
    main()
