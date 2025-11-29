#!/usr/bin/env python3
"""
Image Enhancer - Pipeline de Mejora de Calidad

Este script procesa las imágenes de personas extraídas por video_processor.py
y aplica técnicas de mejora de calidad para facilitar la detección de armas
en el Stage 2 del pipeline.

Técnicas aplicadas:
- Interpolación spline para redimensionamiento (mínimo 200x100 px)
- Reducción de ruido (filtro bilateral)
- Realce de nitidez (unsharp masking)
- Ajuste de contraste adaptativo (CLAHE)
- Realce de bordes
- Mejora de brillo (ajuste canal V en HSV)

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Octubre 2025
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Tuple, Dict
from scipy import ndimage
from tqdm import tqdm

# Configuración por defecto
DEFAULT_INPUT_DIR = 'output/cropped_persons'
DEFAULT_OUTPUT_DIR = 'output/enhanced_persons'
MIN_HEIGHT = 200
MIN_WIDTH = 100

class ImageEnhancer:
    """
    Clase para mejorar la calidad de imágenes de personas extraídas.
    
    Aplica un pipeline de técnicas de procesamiento de imágenes para
    mejorar la resolución, nitidez y calidad general de las imágenes.
    """
    
    def __init__(self, min_height: int = MIN_HEIGHT, min_width: int = MIN_WIDTH):
        """
        Inicializa el mejorador de imágenes.
        
        Args:
            min_height: Altura mínima objetivo en píxeles
            min_width: Ancho mínimo objetivo en píxeles
        """
        self.min_height = min_height
        self.min_width = min_width
        self.stats = {
            'total_processed': 0,
            'upscaled': 0,
            'original_sizes': [],
            'enhanced_sizes': [],
            'failed': 0
        }
    
    def resize_with_spline(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona la imagen usando interpolación spline cúbica.
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen redimensionada con interpolación spline
        """
        h, w = image.shape[:2]
        
        # Calcular nuevo tamaño manteniendo aspect ratio
        if h < self.min_height or w < self.min_width:
            scale_h = self.min_height / h if h < self.min_height else 1.0
            scale_w = self.min_width / w if w < self.min_width else 1.0
            scale = max(scale_h, scale_w)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Interpolación cúbica spline (INTER_CUBIC en OpenCV)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.stats['upscaled'] += 1
            return resized
        
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora el contraste usando CLAHE (Adaptive Histogram Equalization).
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen con contraste mejorado
        """
        # Convertir a LAB para aplicar CLAHE solo en el canal de luminosidad
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L (luminosidad)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Reconstruir imagen
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def enhance_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora el brillo de la imagen aumentando el canal V en HSV.
        Args:
            image: Imagen de entrada (BGR)
        Returns:
            Imagen con brillo mejorado
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        v = np.clip(v, 0, 255)
        hsv_brighter = cv2.merge([h, s, v])
        brighter = cv2.cvtColor(hsv_brighter, cv2.COLOR_HSV2BGR)
        return brighter

    def process_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Aplica el pipeline completo de mejora a una imagen.
        
        Args:
            image_path: Ruta a la imagen de entrada
            
        Returns:
            Tuple con (éxito, diccionario de información)
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                self.stats['failed'] += 1
                return False, {'error': 'No se pudo leer la imagen'}
            
            original_shape = image.shape[:2]
            self.stats['original_sizes'].append(original_shape)
            
            # Pipeline de mejora
            # 1. Redimensionar con spline
            enhanced = self.resize_with_spline(image)

            # 2. Mejorar contraste
            enhanced = self.enhance_contrast(enhanced)
            
            # 3. Mejorar brillo
            enhanced = self.enhance_brightness(enhanced)

            enhanced_shape = enhanced.shape[:2]
            self.stats['enhanced_sizes'].append(enhanced_shape)
            self.stats['total_processed'] += 1

            return True, {
                'original_size': original_shape,
                'enhanced_size': enhanced_shape,
                'enhanced_image': enhanced
            }

        except Exception as e:
            self.stats['failed'] += 1
            return False, {'error': str(e)}
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """
        Procesa todas las imágenes de un directorio.
        
        Args:
            input_dir: Directorio con imágenes de entrada
            output_dir: Directorio para guardar imágenes mejoradas
        """
        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener lista de imágenes
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        if not image_files:
            print(f"⚠️  No se encontraron imágenes en {input_dir}")
            return
        
        print(f"\n🔍 Encontradas {len(image_files)} imágenes para procesar")
        print(f"📐 Tamaño mínimo objetivo: {self.min_height}x{self.min_width} píxeles")
        print(f"\n🚀 Iniciando pipeline de mejora...\n")
        
        # Procesar cada imagen con barra de progreso
        for image_path in tqdm(image_files, desc="Procesando imágenes", unit="img"):
            success, info = self.process_image(image_path)
            
            if success:
                # Guardar imagen mejorada
                output_path = output_dir / image_path.name
                cv2.imwrite(str(output_path), info['enhanced_image'])
            else:
                print(f"\n⚠️  Error procesando {image_path.name}: {info.get('error', 'Desconocido')}")
    
    def print_statistics(self) -> None:
        """
        Imprime estadísticas del procesamiento.
        """
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS DEL PROCESAMIENTO")
        print("=" * 60)
        print(f"✅ Imágenes procesadas exitosamente: {self.stats['total_processed']}")
        print(f"⬆️  Imágenes escaladas (upscaling): {self.stats['upscaled']}")
        print(f"❌ Imágenes fallidas: {self.stats['failed']}")
        
        if self.stats['original_sizes']:
            avg_orig_h = np.mean([s[0] for s in self.stats['original_sizes']])
            avg_orig_w = np.mean([s[1] for s in self.stats['original_sizes']])
            avg_enh_h = np.mean([s[0] for s in self.stats['enhanced_sizes']])
            avg_enh_w = np.mean([s[1] for s in self.stats['enhanced_sizes']])
            
            print(f"\n📏 Tamaños promedio:")
            print(f"   Original:  {avg_orig_h:.0f}x{avg_orig_w:.0f} píxeles")
            print(f"   Mejorado:  {avg_enh_h:.0f}x{avg_enh_w:.0f} píxeles")
            print(f"   Aumento:   {(avg_enh_h/avg_orig_h - 1)*100:.1f}% altura, "
                  f"{(avg_enh_w/avg_orig_w - 1)*100:.1f}% ancho")


def main():
    """
    Función principal del programa.
    """
    print("🎨 Pipeline de Mejora de Calidad de Imágenes")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Mejora la calidad de imágenes de personas extraídas"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directorio con imágenes de entrada (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=MIN_HEIGHT,
        help=f"Altura mínima en píxeles (default: {MIN_HEIGHT})"
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=MIN_WIDTH,
        help=f"Ancho mínimo en píxeles (default: {MIN_WIDTH})"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Verificar que existe el directorio de entrada
    if not input_dir.exists():
        print(f"❌ Error: Directorio de entrada no encontrado: {input_dir}")
        print(f"   Ejecuta primero video_processor.py para extraer personas")
        return 1
    
    # Crear enhancer y procesar
    enhancer = ImageEnhancer(min_height=args.min_height, min_width=args.min_width)
    enhancer.process_directory(input_dir, output_dir)
    
    # Mostrar estadísticas
    enhancer.print_statistics()
    
    print(f"\n✅ Procesamiento completado")
    print(f"📁 Imágenes mejoradas guardadas en: {output_dir}")
    print(f"\n💡 Las imágenes están listas para el Stage 2 (detección de armas)")
    
    return 0


if __name__ == "__main__":
    exit(main())
