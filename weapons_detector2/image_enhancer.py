#!/usr/bin/env python3
"""
Image Enhancer - Versión simplificada para entrenamiento

Aplica mejoras de calidad a imágenes para facilitar la detección de armas.

Técnicas aplicadas:
- Ajuste de contraste adaptativo (CLAHE)
- Mejora de brillo (ajuste canal V en HSV)

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Noviembre 2025
"""

import cv2
import numpy as np


class ImageEnhancer:
    """
    Clase simplificada para mejorar la calidad de imágenes durante el entrenamiento.
    
    Aplica un pipeline de técnicas de procesamiento de imágenes para
    mejorar el contraste y brillo de las imágenes.
    """
    
    def __init__(self):
        """Inicializa el mejorador de imágenes."""
        pass
    
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

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica el pipeline completo de mejora a una imagen.
        
        Args:
            image: Imagen de entrada (BGR/OpenCV format)
            
        Returns:
            Imagen mejorada (BGR/OpenCV format)
        """
        # Pipeline de mejora
        # 1. Mejorar contraste
        enhanced = self.enhance_contrast(image)
        
        # 2. Mejorar brillo
        enhanced = self.enhance_brightness(enhanced)
        
        return enhanced
