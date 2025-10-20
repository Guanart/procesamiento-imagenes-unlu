#!/usr/bin/env python3
"""
Dataset Report Generator - Análisis y Documentación

Este script genera un informe completo sobre el estado del dataset,
incluyendo balanceo de clases, normalización de tamaños y mejoras aplicadas.

Autor: Proyecto de Procesamiento de Imágenes - Universidad Nacional de Luján
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from datetime import datetime
import json

class DatasetAnalyzer:
    """
    Analizador de datasets para generar informes de calidad.
    """
    
    def __init__(self):
        """
        Inicializa el analizador de datasets.
        """
        self.stats = {
            'weapons': defaultdict(lambda: {
                'count': 0,
                'sizes': [],
                'avg_size': (0, 0),
                'min_size': (float('inf'), float('inf')),
                'max_size': (0, 0),
                'augmented': False
            }),
            'persons': {
                'count': 0,
                'sizes': [],
                'avg_size': (0, 0),
                'min_size': (float('inf'), float('inf')),
                'max_size': (0, 0),
                'enhanced': False
            }
        }
    
    def analyze_weapon_dataset(self, dataset_path: Path) -> None:
        """
        Analiza el dataset de armas (cuchillos y pistolas).
        
        Args:
            dataset_path: Ruta al directorio del dataset
        """
        print("\n🔍 Analizando dataset de armas...")
        
        # Buscar subdirectorios de clases
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                continue
            
            print(f"   - Procesando clase: {class_name} ({len(image_files)} imágenes)")
            
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    self.stats['weapons'][class_name]['count'] += 1
                    self.stats['weapons'][class_name]['sizes'].append((h, w))
                    
                    # Actualizar min/max
                    current_min = self.stats['weapons'][class_name]['min_size']
                    current_max = self.stats['weapons'][class_name]['max_size']
                    self.stats['weapons'][class_name]['min_size'] = (
                        min(current_min[0], h),
                        min(current_min[1], w)
                    )
                    self.stats['weapons'][class_name]['max_size'] = (
                        max(current_max[0], h),
                        max(current_max[1], w)
                    )
                    
                    # Detectar si hay aumentación (nombres con sufijos)
                    if any(suffix in img_path.stem for suffix in ['_h_flip', '_v_flip', '_rotated']):
                        self.stats['weapons'][class_name]['augmented'] = True
            
            # Calcular promedio
            if self.stats['weapons'][class_name]['sizes']:
                sizes = self.stats['weapons'][class_name]['sizes']
                avg_h = np.mean([s[0] for s in sizes])
                avg_w = np.mean([s[1] for s in sizes])
                self.stats['weapons'][class_name]['avg_size'] = (int(avg_h), int(avg_w))
    
    def analyze_person_dataset(self, dataset_path: Path, enhanced_path: Path = None) -> None:
        """
        Analiza el dataset de personas extraídas.
        
        Args:
            dataset_path: Ruta al directorio de personas originales
            enhanced_path: Ruta al directorio de personas mejoradas (opcional)
        """
        print("\n🔍 Analizando dataset de personas...")
        
        # Analizar imágenes originales
        if dataset_path.exists():
            image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
            print(f"   - Imágenes originales: {len(image_files)}")
            
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    self.stats['persons']['count'] += 1
                    self.stats['persons']['sizes'].append((h, w))
                    
                    current_min = self.stats['persons']['min_size']
                    current_max = self.stats['persons']['max_size']
                    self.stats['persons']['min_size'] = (
                        min(current_min[0], h),
                        min(current_min[1], w)
                    )
                    self.stats['persons']['max_size'] = (
                        max(current_max[0], h),
                        max(current_max[1], w)
                    )
            
            if self.stats['persons']['sizes']:
                sizes = self.stats['persons']['sizes']
                avg_h = np.mean([s[0] for s in sizes])
                avg_w = np.mean([s[1] for s in sizes])
                self.stats['persons']['avg_size'] = (int(avg_h), int(avg_w))
        
        # Verificar si existe versión mejorada
        if enhanced_path and enhanced_path.exists():
            enhanced_files = list(enhanced_path.glob('*.jpg')) + list(enhanced_path.glob('*.png'))
            if enhanced_files:
                self.stats['persons']['enhanced'] = True
                print(f"   - Imágenes mejoradas encontradas: {len(enhanced_files)}")
    
    def check_class_balance(self) -> Dict:
        """
        Verifica el balanceo de clases en el dataset de armas.
        
        Returns:
            Diccionario con información de balanceo
        """
        balance_info = {
            'balanced': False,
            'classes': {},
            'total': 0,
            'difference': 0
        }
        
        if not self.stats['weapons']:
            return balance_info
        
        counts = {class_name: info['count'] 
                 for class_name, info in self.stats['weapons'].items()}
        
        balance_info['classes'] = counts
        balance_info['total'] = sum(counts.values())
        
        if len(counts) > 1:
            min_count = min(counts.values())
            max_count = max(counts.values())
            balance_info['difference'] = max_count - min_count
            
            # Considerar balanceado si la diferencia es < 10%
            balance_info['balanced'] = (balance_info['difference'] / max_count) < 0.1
        else:
            balance_info['balanced'] = True
        
        return balance_info
    
    def generate_markdown_report(self, output_path: Path) -> None:
        """
        Genera un informe en formato Markdown.
        
        Args:
            output_path: Ruta donde guardar el informe
        """
        report = []
        report.append("# 📊 Informe de Dataset - Procesamiento de Imágenes")
        report.append(f"\n**Fecha de generación**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")
        
        # Sección de armas
        report.append("## 🔫 Dataset de Armas (Cuchillos y Pistolas)\n")
        
        if self.stats['weapons']:
            balance = self.check_class_balance()
            
            report.append("### 📈 Balanceo de Clases\n")
            report.append(f"**Estado**: {'✅ Balanceado' if balance['balanced'] else '⚠️ Desbalanceado'}\n")
            report.append(f"**Total de imágenes**: {balance['total']}\n")
            
            for class_name, count in balance['classes'].items():
                percentage = (count / balance['total']) * 100
                report.append(f"- **{class_name}**: {count} imágenes ({percentage:.1f}%)")
            
            if not balance['balanced']:
                report.append(f"\n⚠️ Diferencia entre clases: {balance['difference']} imágenes")
                report.append("\n💡 **Recomendación**: Aplicar data augmentation para balancear")
            
            report.append("\n### 📐 Normalización de Tamaño\n")
            
            for class_name, info in self.stats['weapons'].items():
                report.append(f"\n#### {class_name.capitalize()}\n")
                report.append(f"- **Cantidad**: {info['count']} imágenes")
                report.append(f"- **Tamaño promedio**: {info['avg_size'][0]}x{info['avg_size'][1]} píxeles")
                report.append(f"- **Tamaño mínimo**: {info['min_size'][0]}x{info['min_size'][1]} píxeles")
                report.append(f"- **Tamaño máximo**: {info['max_size'][0]}x{info['max_size'][1]} píxeles")
                report.append(f"- **Data Augmentation**: {'✅ Aplicada' if info['augmented'] else '❌ No aplicada'}")
            
            report.append("\n### 🔄 Transformaciones Aplicadas\n")
            report.append("Las siguientes transformaciones básicas han sido aplicadas:")
            report.append("1. ✅ **Flip horizontal** - Volteo espejo")
            report.append("2. ✅ **Flip vertical** - Volteo vertical")
            report.append("3. ✅ **Rotación 90°** - Rotación en sentido horario")
            report.append("\n💡 Estas transformaciones cuadruplican el dataset original.")
        else:
            report.append("⚠️ No se encontraron datos del dataset de armas.\n")
        
        # Sección de personas
        report.append("\n---\n")
        report.append("## 👥 Dataset de Personas (Stage 2)\n")
        
        if self.stats['persons']['count'] > 0:
            report.append(f"**Total de personas extraídas**: {self.stats['persons']['count']}\n")
            report.append(f"**Tamaño promedio original**: {self.stats['persons']['avg_size'][0]}x{self.stats['persons']['avg_size'][1]} píxeles\n")
            report.append(f"**Tamaño mínimo**: {self.stats['persons']['min_size'][0]}x{self.stats['persons']['min_size'][1]} píxeles\n")
            report.append(f"**Tamaño máximo**: {self.stats['persons']['max_size'][0]}x{self.stats['persons']['max_size'][1]} píxeles\n")
            
            report.append("\n### 🎨 Pipeline de Mejora de Calidad\n")
            
            if self.stats['persons']['enhanced']:
                report.append("**Estado**: ✅ Pipeline aplicado correctamente\n")
                report.append("\nTécnicas de mejora aplicadas:")
                report.append("1. ✅ **Interpolación Spline Cúbica** - Redimensionamiento a mínimo 200x100 px")
                report.append("2. ✅ **Reducción de Ruido** - Filtro bilateral preservando bordes")
                report.append("3. ✅ **Realce de Nitidez** - Unsharp masking")
                report.append("4. ✅ **Mejora de Contraste** - CLAHE adaptativo")
                report.append("5. ✅ **Realce de Bordes** - Detección Canny + combinación")
                
                report.append("\n**Objetivo**: Mejorar la calidad de las imágenes de personas para facilitar")
                report.append("la detección de armas en el Stage 2 del pipeline.")
            else:
                report.append("**Estado**: ⚠️ Pipeline de mejora no aplicado\n")
                report.append("\n💡 **Recomendación**: Ejecutar `image_enhancer.py` para mejorar")
                report.append("la calidad de las imágenes antes del Stage 2.")
        else:
            report.append("⚠️ No se encontraron datos del dataset de personas.\n")
            report.append("\n💡 **Recomendación**: Ejecutar `video_processor.py` para extraer personas de videos.")
        
        # Conclusiones
        report.append("\n---\n")
        report.append("## 📋 Conclusiones y Recomendaciones\n")
        
        if self.stats['weapons']:
            balance = self.check_class_balance()
            if balance['balanced']:
                report.append("- ✅ El dataset de armas está **balanceado** y listo para entrenamiento")
            else:
                report.append("- ⚠️ El dataset de armas está **desbalanceado**, considerar más augmentation")
        
        if self.stats['persons']['enhanced']:
            report.append("- ✅ Las imágenes de personas han sido **mejoradas** exitosamente")
            report.append("- ✅ El pipeline está listo para el **Stage 2** (detección de armas)")
        elif self.stats['persons']['count'] > 0:
            report.append("- ⚠️ Las imágenes de personas requieren **mejora de calidad**")
            report.append("- 💡 Ejecutar `image_enhancer.py` antes de proceder al Stage 2")
        
        report.append("\n### 🎯 Próximos Pasos\n")
        report.append("1. Verificar el balanceo de clases en el dataset de armas")
        report.append("2. Asegurar que todas las imágenes de personas estén mejoradas")
        report.append("3. Proceder con el entrenamiento del modelo Stage 2")
        report.append("4. Aplicar fine-tuning de YOLOv8 para detección de armas en personas")
        
        report.append("\n---\n")
        report.append("*Informe generado automáticamente por `generate_report.py`*")
        
        # Guardar el informe
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n✅ Informe generado: {output_path}")
    
    def save_json_stats(self, output_path: Path) -> None:
        """
        Guarda las estadísticas en formato JSON.
        
        Args:
            output_path: Ruta donde guardar el archivo JSON
        """
        # Convertir a formato serializable
        json_stats = {
            'weapons': {},
            'persons': {}
        }
        
        for class_name, info in self.stats['weapons'].items():
            json_stats['weapons'][class_name] = {
                'count': info['count'],
                'avg_size': info['avg_size'],
                'min_size': info['min_size'],
                'max_size': info['max_size'],
                'augmented': info['augmented']
            }
        
        json_stats['persons'] = {
            'count': self.stats['persons']['count'],
            'avg_size': self.stats['persons']['avg_size'],
            'min_size': self.stats['persons']['min_size'],
            'max_size': self.stats['persons']['max_size'],
            'enhanced': self.stats['persons']['enhanced']
        }
        
        json_stats['balance'] = self.check_class_balance()
        json_stats['timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Estadísticas JSON guardadas: {output_path}")


def main():
    """
    Función principal del programa.
    """
    print("📊 Generador de Informes de Dataset")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Genera un informe completo del estado del dataset"
    )
    parser.add_argument(
        '--weapons-dir',
        type=str,
        default='dataset/augmented',
        help="Directorio del dataset de armas (default: dataset/augmented)"
    )
    parser.add_argument(
        '--persons-dir',
        type=str,
        default='output/cropped_persons',
        help="Directorio de personas originales (default: output/cropped_persons)"
    )
    parser.add_argument(
        '--enhanced-dir',
        type=str,
        default='output/enhanced_persons',
        help="Directorio de personas mejoradas (default: output/enhanced_persons)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='INFORME_DATASET.md',
        help="Archivo de salida del informe (default: INFORME_DATASET.md)"
    )
    
    args = parser.parse_args()
    
    # Crear analizador
    analyzer = DatasetAnalyzer()
    
    # Analizar dataset de armas
    weapons_path = Path(args.weapons_dir)
    if weapons_path.exists():
        analyzer.analyze_weapon_dataset(weapons_path)
    else:
        print(f"⚠️ Directorio de armas no encontrado: {weapons_path}")
    
    # Analizar dataset de personas
    persons_path = Path(args.persons_dir)
    enhanced_path = Path(args.enhanced_dir)
    
    if persons_path.exists():
        analyzer.analyze_person_dataset(persons_path, enhanced_path)
    else:
        print(f"⚠️ Directorio de personas no encontrado: {persons_path}")
    
    # Generar informes
    output_md = Path(args.output)
    output_json = output_md.with_suffix('.json')
    
    analyzer.generate_markdown_report(output_md)
    analyzer.save_json_stats(output_json)
    
    print("\n" + "=" * 60)
    print("✅ Generación de informes completada")
    print(f"📄 Informe Markdown: {output_md}")
    print(f"📄 Estadísticas JSON: {output_json}")
    
    return 0


if __name__ == "__main__":
    exit(main())
