#!/usr/bin/env python3
"""
Data Augmentation para Dataset de Detección de Armas
Genera imágenes aumentadas con sus correspondientes anotaciones XML (Pascal VOC).

Transformaciones aplicadas:
- Flip horizontal (50%)
- Rotaciones ligeras (-15° a +15°)
- Ajustes de brillo (0.7 - 1.3)
- Ajustes de contraste (0.7 - 1.3)
- Ajustes de saturación (0.7 - 1.3)
- Blur gaussiano ocasional (10%)
- Ruido gaussiano ocasional (10%)
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import argparse
from tqdm import tqdm
from typing import List, Tuple
import shutil


class DataAugmentor:
    """Genera versiones aumentadas de imágenes con sus anotaciones XML."""
    
    # Extensiones válidas para buscar imágenes
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        saturation_range: Tuple[float, float] = (0.7, 1.3),
        blur_prob: float = 0.1,
        noise_prob: float = 0.1
    ):
        """
        Inicializa el augmentor.
        
        Args:
            flip_prob: Probabilidad de flip horizontal
            rotate_prob: Probabilidad de rotación
            brightness_range: Rango de ajuste de brillo
            contrast_range: Rango de ajuste de contraste
            saturation_range: Rango de ajuste de saturación
            blur_prob: Probabilidad de aplicar blur
            noise_prob: Probabilidad de aplicar ruido
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
    
    def _resolve_image(self, filename: str, images_dir: Path) -> Path:
        """
        Busca la imagen con diferentes extensiones.
        
        Args:
            filename: Nombre del archivo (puede no tener la extensión correcta)
            images_dir: Directorio donde buscar
            
        Returns:
            Path de la imagen encontrada o None
        """
        base = Path(filename)
        for ext in self.VALID_EXTS:
            cand = images_dir / (base.stem + ext)
            if cand.exists():
                return cand
        return None
    
    def flip_horizontal(self, image: np.ndarray, boxes: List[Tuple]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Aplica flip horizontal a imagen y ajusta bounding boxes.
        
        Args:
            image: Imagen BGR
            boxes: Lista de (label, xmin, ymin, xmax, ymax)
            
        Returns:
            Tupla (imagen_flippeada, boxes_ajustadas)
        """
        flipped = cv2.flip(image, 1)
        height, width = image.shape[:2]
        
        new_boxes = []
        for label, xmin, ymin, xmax, ymax in boxes:
            # Invertir coordenadas X
            new_xmin = width - xmax
            new_xmax = width - xmin
            new_boxes.append((label, new_xmin, ymin, new_xmax, ymax))
        
        return flipped, new_boxes
    
    def rotate_image(self, image: np.ndarray, boxes: List[Tuple], angle: float) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Rota imagen y ajusta bounding boxes.
        
        Args:
            image: Imagen BGR
            boxes: Lista de (label, xmin, ymin, xmax, ymax)
            angle: Ángulo de rotación en grados
            
        Returns:
            Tupla (imagen_rotada, boxes_ajustadas)
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotar imagen
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Ajustar bounding boxes
        new_boxes = []
        for label, xmin, ymin, xmax, ymax in boxes:
            # Rotar las 4 esquinas del bounding box
            corners = np.array([
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmax, ymax, 1],
                [xmin, ymax, 1]
            ])
            
            rotated_corners = rotation_matrix @ corners.T
            
            # Calcular nuevo bounding box
            new_xmin = int(np.min(rotated_corners[0]))
            new_xmax = int(np.max(rotated_corners[0]))
            new_ymin = int(np.min(rotated_corners[1]))
            new_ymax = int(np.max(rotated_corners[1]))
            
            # Asegurar que están dentro de la imagen
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(width, new_xmax)
            new_ymax = min(height, new_ymax)
            
            # Solo agregar si el box sigue siendo válido
            if new_xmax > new_xmin and new_ymax > new_ymin:
                new_boxes.append((label, new_xmin, new_ymin, new_xmax, new_ymax))
        
        return rotated, new_boxes
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajusta el brillo de la imagen.
        
        Convierte a espacio HSV (Hue-Saturation-Value) para modificar solo el canal V (brillo)
        sin afectar los colores. Esto simula diferentes condiciones de iluminación.
        
        Args:
            image: Imagen BGR
            factor: Multiplicador de brillo (0.7-1.3). <1 oscurece, >1 aclara
        """
        # Convertir el espacio de color BGR a HSV, para acceder al canal de brillo (V)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Multiplicar solo el canal V (índice 2) por el factor
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        # Asegurar que los valores estén en rango válido [0, 255]
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        # Volver a BGR para compatibilidad con OpenCV
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajusta el contraste de la imagen.
        
        Aplica una transformación lineal alrededor de la media de la imagen.
        Aumentar contraste enfatiza diferencias entre píxeles claros y oscuros,
        lo que ayuda al modelo a detectar bordes y formas de armas en condiciones variables.
        
        Args:
            image: Imagen BGR
            factor: Multiplicador de contraste (0.8-1.2). <1 reduce contraste, >1 lo aumenta
        """
        # Calcular valor medio de todos los píxeles
        mean = np.mean(image)
        # Fórmula: I_out = (I_in - mean) * factor + mean
        # Expande/contrae los valores alrededor de la media
        adjusted = (image - mean) * factor + mean
        # Asegurar rango válido [0, 255]
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajusta la saturación de la imagen.
        
        Modifica la intensidad de color en el espacio HSV (canal S - Saturation).
        Variar saturación simula diferentes condiciones de captura (cámaras, sensores)
        y ayuda al modelo a generalizar entre imágenes vívidas y desaturadas.
        
        Args:
            image: Imagen BGR
            factor: Multiplicador de saturación (0.7-1.3). <1 desatura, >1 intensifica colores
        """
        # Convertir BGR -> HSV para acceder al canal de saturación (S)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Multiplicar solo el canal S (índice 1) por el factor
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        # Asegurar que los valores estén en rango válido [0, 255]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        # Volver a BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Aplica blur gaussiano.
        
        Simula desenfoque por movimiento, distancia focal incorrecta o cámaras de baja calidad.
        Esto fuerza al modelo a aprender características robustas que no dependen de bordes nítidos.
        
        Args:
            image: Imagen BGR
        
        Returns:
            Imagen con desenfoque gaussiano (kernel 3x3 o 5x5 aleatorio)
        """
        # Elegir tamaño de kernel aleatoriamente (3x3 es suave, 5x5 es más fuerte)
        kernel_size = random.choice([3, 5])
        # Aplicar filtro gaussiano con sigma=0 (calculado automáticamente)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Aplica ruido gaussiano.
        
        Simula artefactos de sensores (ISO alto, poca luz, compresión JPEG).
        El ruido obliga al modelo a distinguir señales importantes (armas) del ruido de fondo.
        
        Args:
            image: Imagen BGR
        
        Returns:
            Imagen con ruido gaussiano aditivo (media=0, desviación estándar=10)
        """
        # Generar ruido gaussiano con distribución N(0, 10)
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        # Sumar ruido a la imagen (aditivo)
        noisy = cv2.add(image, noise)
        return noisy
    
    def augment_image(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple],
        num_augmentations: int = 2
    ) -> List[Tuple[np.ndarray, List[Tuple], str]]:
        """
        Genera múltiples versiones aumentadas de una imagen.
        
        Args:
            image: Imagen BGR
            boxes: Lista de (label, xmin, ymin, xmax, ymax)
            num_augmentations: Número de versiones a generar
            
        Returns:
            Lista de tuplas (imagen_aumentada, boxes_ajustadas, sufijo_nombre)
        """
        augmented = []
        
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_boxes = boxes.copy()
            suffix_parts = []
            
            # Flip horizontal
            if random.random() < self.flip_prob:
                aug_image, aug_boxes = self.flip_horizontal(aug_image, aug_boxes)
                suffix_parts.append("flip")
            
            # Rotación
            if random.random() < self.rotate_prob:
                angle = random.uniform(-15, 15)
                aug_image, aug_boxes = self.rotate_image(aug_image, aug_boxes, angle)
                suffix_parts.append(f"rot{int(angle)}")
            
            # Ajustes de color (siempre aplicar al menos uno)
            brightness_factor = random.uniform(*self.brightness_range)
            aug_image = self.adjust_brightness(aug_image, brightness_factor)
            
            contrast_factor = random.uniform(*self.contrast_range)
            aug_image = self.adjust_contrast(aug_image, contrast_factor)
            
            saturation_factor = random.uniform(*self.saturation_range)
            aug_image = self.adjust_saturation(aug_image, saturation_factor)
            
            suffix_parts.append("col")
            
            # Blur
            if random.random() < self.blur_prob:
                aug_image = self.apply_blur(aug_image)
                suffix_parts.append("blur")
            
            # Ruido
            if random.random() < self.noise_prob:
                aug_image = self.apply_noise(aug_image)
                suffix_parts.append("noise")
            
            suffix = "_".join(suffix_parts) + f"_v{i+1}"
            augmented.append((aug_image, aug_boxes, suffix))
        
        return augmented
    
    def parse_xml(self, xml_path: Path) -> Tuple[str, List[Tuple]]:
        """
        Lee un archivo XML y extrae las anotaciones.
        
        Args:
            xml_path: Ruta al archivo XML
            
        Returns:
            Tupla (filename, lista de boxes)
        """
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        
        filename = root.find('filename').text
        
        boxes = []
        for obj in root.findall('object'):
            label = obj.find('name').text.strip().lower()
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append((label, xmin, ymin, xmax, ymax))
        
        return filename, boxes
    
    def create_xml(
        self,
        filename: str,
        width: int,
        height: int,
        boxes: List[Tuple],
        output_path: Path
    ):
        """
        Crea un archivo XML con las anotaciones.
        
        Args:
            filename: Nombre del archivo de imagen
            width: Ancho de la imagen
            height: Alto de la imagen
            boxes: Lista de (label, xmin, ymin, xmax, ymax)
            output_path: Ruta donde guardar el XML
        """
        annotation = ET.Element('annotation')
        
        ET.SubElement(annotation, 'folder').text = 'augmented'
        ET.SubElement(annotation, 'filename').text = filename
        
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        for label, xmin, ymin, xmax, ymax in boxes:
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = label
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(int(xmin))
            ET.SubElement(bbox, 'ymin').text = str(int(ymin))
            ET.SubElement(bbox, 'xmax').text = str(int(xmax))
            ET.SubElement(bbox, 'ymax').text = str(int(ymax))
        
        tree = ET.ElementTree(annotation)
        ET.indent(tree, space='  ')
        tree.write(str(output_path), encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(
        description="Data Augmentation para dataset de detección de armas"
    )
    parser.add_argument(
        '--images-dir',
        default='dataset/images',
        help='Directorio con imágenes originales'
    )
    parser.add_argument(
        '--xml-dir',
        default='dataset/xmls',
        help='Directorio con XMLs originales'
    )
    parser.add_argument(
        '--output-images-dir',
        default='dataset_augmented/images',
        help='Directorio de salida para imágenes aumentadas'
    )
    parser.add_argument(
        '--output-xml-dir',
        default='dataset_augmented/xmls',
        help='Directorio de salida para XMLs aumentados'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=2,
        help='Número de versiones aumentadas por imagen'
    )
    parser.add_argument(
        '--copy-originals',
        action='store_true',
        help='Copiar también las imágenes originales al dataset aumentado'
    )
    
    args = parser.parse_args()
    
    # Crear directorios de salida
    output_images_dir = Path(args.output_images_dir)
    output_xml_dir = Path(args.output_xml_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_xml_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎨 DATA AUGMENTATION - Dataset de Detección de Armas")
    print("=" * 70)
    print(f"📂 Imágenes de entrada: {args.images_dir}")
    print(f"📂 XMLs de entrada: {args.xml_dir}")
    print(f"📂 Imágenes de salida: {args.output_images_dir}")
    print(f"📂 XMLs de salida: {args.output_xml_dir}")
    print(f"🔢 Augmentaciones por imagen: {args.num_augmentations}")
    print(f"📋 Copiar originales: {'Sí' if args.copy_originals else 'No'}")
    print("=" * 70)
    
    # Inicializar augmentor
    augmentor = DataAugmentor()
    
    # Obtener lista de XMLs
    xml_files = list(Path(args.xml_dir).glob('*.xml'))
    
    if not xml_files:
        print(f"❌ No se encontraron archivos XML en {args.xml_dir}")
        return 1
    
    print(f"\n🔍 Encontrados {len(xml_files)} archivos XML")
    
    # Estadísticas
    total_original = len(xml_files)
    total_generated = 0
    failed = 0
    
    # Procesar cada XML
    print("\n🚀 Iniciando augmentation...\n")
    
    for xml_path in tqdm(xml_files, desc="Procesando imágenes", unit="img"):
        try:
            # Parsear XML
            filename, boxes = augmentor.parse_xml(xml_path)
            
            if not boxes:
                print(f"⚠️  Saltando {filename}: sin anotaciones")
                continue
            
            # Buscar imagen con diferentes extensiones (como train_fasterrcnn_light.py)
            image_path = augmentor._resolve_image(filename, Path(args.images_dir))
            if image_path is None:
                print(f"⚠️  Imagen no encontrada: {filename}")
                failed += 1
                continue
            
            # Usar la extensión real de la imagen encontrada
            actual_filename = image_path.name
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️  Error al leer imagen: {actual_filename}")
                failed += 1
                continue
            
            height, width = image.shape[:2]
            
            # Copiar original si se requiere
            if args.copy_originals:
                output_image_path = output_images_dir / actual_filename
                output_xml_path = output_xml_dir / xml_path.name
                shutil.copy(image_path, output_image_path)
                shutil.copy(xml_path, output_xml_path)
            
            # Generar versiones aumentadas
            augmented = augmentor.augment_image(image, boxes, args.num_augmentations)
            
            for aug_image, aug_boxes, suffix in augmented:
                # Nombres de archivos aumentados (usar extensión real)
                stem = Path(actual_filename).stem
                ext = Path(actual_filename).suffix
                aug_filename = f"{stem}_{suffix}{ext}"
                aug_xml_name = f"{stem}_{suffix}.xml"
                
                # Guardar imagen aumentada
                aug_image_path = output_images_dir / aug_filename
                cv2.imwrite(str(aug_image_path), aug_image)
                
                # Guardar XML aumentado
                aug_xml_path = output_xml_dir / aug_xml_name
                aug_height, aug_width = aug_image.shape[:2]
                augmentor.create_xml(aug_filename, aug_width, aug_height, aug_boxes, aug_xml_path)
                
                total_generated += 1
        
        except Exception as e:
            print(f"❌ Error procesando {xml_path.name}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("📊 RESUMEN DEL AUGMENTATION")
    print("=" * 70)
    print(f"✅ Imágenes originales: {total_original}")
    print(f"✅ Imágenes aumentadas generadas: {total_generated}")
    print(f"✅ Total en dataset: {total_original + total_generated if args.copy_originals else total_generated}")
    print(f"❌ Errores: {failed}")
    print(f"\n💾 Dataset aumentado guardado en:")
    print(f"   📁 {output_images_dir}")
    print(f"   📁 {output_xml_dir}")
    print("=" * 70)
    
    print("\n💡 Siguiente paso:")
    print(f"   python src/weapon_detection/training/train_fasterrcnn_light.py \\")
    print(f"     --images-dir {args.output_images_dir} \\")
    print(f"     --xml-dir {args.output_xml_dir} \\")
    print(f"     --epochs 50 --batch-size 8")
    
    return 0


if __name__ == "__main__":
    exit(main())
