#!/usr/bin/env python3
"""
Image Enhancer BASIC - Pipeline con Técnicas Básicas de Teoría

Este script es una versión conservadora que usa SOLO técnicas explícitamente
cubiertas en la teoría del curso de Procesamiento de Imágenes.

Técnicas aplicadas (TODAS de teoría básica):
- Interpolación bicúbica (spline) para redimensionamiento ✅
- Filtro Gaussiano para reducción de ruido ✅
- Filtro Laplaciano para realce de nitidez ✅
- Ecualización de Histograma para mejora de contraste ✅
- Operador Sobel para detección de bordes ✅

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Dict

# Configuración por defecto
DEFAULT_INPUT_DIR = 'output/cropped_persons'
DEFAULT_OUTPUT_DIR = 'output/enhanced_persons_basic'
MIN_HEIGHT = 200
MIN_WIDTH = 100

class BasicImageEnhancer:
    """
    Clase para mejorar imágenes usando SOLO técnicas de teoría básica.
    
    Todas las técnicas están garantizadas en los cursos estándar de
    procesamiento digital de imágenes.
    """
    
    def __init__(self, min_height: int = MIN_HEIGHT, min_width: int = MIN_WIDTH):
        """
        Inicializa el mejorador básico de imágenes.
        
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
    
    def resize_with_cubic_interpolation(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona usando interpolación bicúbica (spline cúbico).
        
        TEORÍA: T3C_Interpolacion_imagen.pdf
        La interpolación bicúbica es una técnica fundamental para
        redimensionamiento con calidad.
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen redimensionada
        """
        h, w = image.shape[:2]
        
        # Calcular nuevo tamaño manteniendo aspect ratio
        if h < self.min_height or w < self.min_width:
            scale_h = self.min_height / h if h < self.min_height else 1.0
            scale_w = self.min_width / w if w < self.min_width else 1.0
            scale = max(scale_h, scale_w)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Interpolación bicúbica (implementación de spline cúbico)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.stats['upscaled'] += 1
            return resized
        
        return image
    
    def gaussian_smoothing(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Aplica filtro Gaussiano para reducir ruido.
        
        TEORÍA: Módulos 3-4 (Filtros Espaciales)
        El filtro Gaussiano es un filtro de suavizado fundamental que
        reduce el ruido preservando mejor los bordes que el filtro de media.
        
        Fórmula del kernel Gaussiano 2D:
        G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        
        Args:
            image: Imagen de entrada (BGR)
            kernel_size: Tamaño del kernel (debe ser impar)
            
        Returns:
            Imagen suavizada
        """
        # Filtro Gaussiano 2D con sigma automático
        smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return smoothed
    
    def laplacian_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Realce de nitidez usando el operador Laplaciano.
        
        TEORÍA: Módulos 4-5 (Filtros de Realce)
        El Laplaciano es un operador de segunda derivada que detecta
        cambios rápidos de intensidad (bordes).
        
        Fórmula: sharp = original + c × Laplaciano
        donde c es un peso para controlar el realce
        
        Kernel Laplaciano típico:
        [[ 0 -1  0]
         [-1  4 -1]
         [ 0 -1  0]]
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen con nitidez realzada
        """
        # Convertir a escala de grises para aplicar Laplaciano
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Convertir Laplaciano a BGR para combinarlo
        laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        
        # Combinar: imagen_original + peso × laplaciano
        # Peso de 0.3 para no sobre-realzar
        sharpened = cv2.addWeighted(image, 1.0, laplacian_bgr, 0.3, 0)
        
        return sharpened
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora el contraste mediante ecualización de histograma.
        
        TEORÍA: Módulos 3-4 (Mejora de Contraste)
        La ecualización de histograma redistribuye las intensidades
        para usar todo el rango dinámico [0, 255].
        
        Algoritmo:
        1. Calcular histograma de la imagen
        2. Calcular función de distribución acumulativa (CDF)
        3. Normalizar CDF al rango [0, 255]
        4. Mapear intensidades originales usando CDF normalizada
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen con contraste mejorado
        """
        # Convertir a YCrCb para ecualizar solo la luminancia
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Ecualizar el canal Y (luminancia)
        y_eq = cv2.equalizeHist(y)
        
        # Reconstruir imagen
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        equalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        
        return equalized
    
    def sobel_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Realza bordes usando el operador Sobel.
        
        TEORÍA: Módulos 4-5 (Detección de Bordes)
        Sobel es un operador de primera derivada que detecta gradientes
        de intensidad (bordes) en direcciones x e y.
        
        Kernels Sobel:
        Gx = [[-1 0 1]      Gy = [[-1 -2 -1]
              [-2 0 2]             [ 0  0  0]
              [-1 0 1]]            [ 1  2  1]]
        
        Magnitud del gradiente: G = √(Gx² + Gy²)
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen con bordes realzados
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar Sobel en x e y
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular magnitud del gradiente
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
        
        # Convertir a BGR
        sobel_bgr = cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
        
        # Combinar con imagen original (peso pequeño)
        enhanced = cv2.addWeighted(image, 0.92, sobel_bgr, 0.08, 0)
        
        return enhanced
    
    def process_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Aplica el pipeline completo de mejora básica a una imagen.
        
        Pipeline:
        1. Interpolación bicúbica (redimensionar)
        2. Filtro Gaussiano (reducir ruido)
        3. Laplaciano (realzar nitidez)
        4. Ecualización de histograma (mejorar contraste)
        5. Sobel (realzar bordes)
        
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
            
            # Pipeline de mejora con técnicas básicas
            # 1. Redimensionar con interpolación bicúbica
            enhanced = self.resize_with_cubic_interpolation(image)
            
            # 2. Reducir ruido con Gaussiano
            enhanced = self.gaussian_smoothing(enhanced, kernel_size=5)
            
            # 3. Realzar nitidez con Laplaciano
            enhanced = self.laplacian_sharpening(enhanced)
            
            # 4. Mejorar contraste con ecualización de histograma
            enhanced = self.histogram_equalization(enhanced)
            
            # 5. Realzar bordes con Sobel
            enhanced = self.sobel_edge_enhancement(enhanced)
            
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
        print(f"📚 Usando SOLO técnicas de teoría básica")
        print(f"\n🚀 Iniciando pipeline básico...\n")
        
        # Procesar cada imagen
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Procesando: {image_path.name}", end=" ... ")
            
            success, info = self.process_image(image_path)
            
            if success:
                # Guardar imagen mejorada
                output_path = output_dir / image_path.name
                cv2.imwrite(str(output_path), info['enhanced_image'])
                print("✅")
            else:
                print(f"❌ Error: {info.get('error', 'Desconocido')}")
    
    def print_statistics(self) -> None:
        """
        Imprime estadísticas del procesamiento.
        """
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS DEL PROCESAMIENTO (VERSIÓN BÁSICA)")
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
        
        print("\n📚 Técnicas aplicadas (TODAS de teoría básica):")
        print("   1. ✅ Interpolación Bicúbica (spline)")
        print("   2. ✅ Filtro Gaussiano")
        print("   3. ✅ Realce con Laplaciano")
        print("   4. ✅ Ecualización de Histograma")
        print("   5. ✅ Detección de bordes con Sobel")


def main():
    """
    Función principal del programa.
    """
    print("🎨 Pipeline Básico de Mejora de Calidad")
    print("📚 Versión con técnicas de teoría estándar")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Mejora imágenes usando SOLO técnicas de teoría básica"
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
        print("   Ejecuta primero video_processor.py para extraer personas")
        return 1
    
    # Crear enhancer y procesar
    enhancer = BasicImageEnhancer(min_height=args.min_height, min_width=args.min_width)
    enhancer.process_directory(input_dir, output_dir)
    
    # Mostrar estadísticas
    enhancer.print_statistics()
    
    print(f"\n✅ Procesamiento completado")
    print(f"📁 Imágenes mejoradas guardadas en: {output_dir}")
    print(f"\n💡 Las imágenes están listas para el Stage 2 (detección de armas)")
    print(f"📚 Todas las técnicas usadas están en la teoría del curso")
    
    return 0


if __name__ == "__main__":
    exit(main())
