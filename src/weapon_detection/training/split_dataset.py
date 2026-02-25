#!/usr/bin/env python3
"""
Dataset Splitter - Separa dataset en training y testing

Este script separa un porcentaje del dataset original para testing,
moviendo las imágenes y XMLs a un directorio separado.
"""

import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def split_dataset(
    images_dir: Path,
    xml_dir: Path,
    test_images_dir: Path,
    test_xml_dir: Path,
    test_split: float = 0.1,
    seed: int = 42
):
    """
    Separa dataset en training y testing.
    
    Args:
        images_dir: Directorio con imágenes originales
        xml_dir: Directorio con XMLs originales
        test_images_dir: Directorio de salida para imágenes de test
        test_xml_dir: Directorio de salida para XMLs de test
        test_split: Fracción del dataset para testing (default: 0.1 = 10%)
        seed: Semilla para reproducibilidad
    """
    # Crear directorios de test
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_xml_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener lista de XMLs
    xml_files = sorted(xml_dir.glob('*.xml'))
    
    if not xml_files:
        print(f"❌ No se encontraron archivos XML en {xml_dir}")
        return
    
    # Configurar seed para reproducibilidad
    random.seed(seed)
    random.shuffle(xml_files)
    
    # Calcular split
    num_test = int(len(xml_files) * test_split)
    test_xmls = xml_files[:num_test]
    
    print("=" * 70)
    print("📊 SPLIT DEL DATASET")
    print("=" * 70)
    print(f"📂 Dataset original: {len(xml_files)} muestras")
    print(f"🧪 Test set: {num_test} muestras ({test_split * 100:.1f}%)")
    print(f"🏋️  Training set: {len(xml_files) - num_test} muestras ({(1 - test_split) * 100:.1f}%)")
    print("=" * 70)
    
    # Extensiones válidas para imágenes
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    moved_count = 0
    failed_count = 0
    
    print("\n🚀 Moviendo archivos al conjunto de test...\n")
    
    for xml_path in tqdm(test_xmls, desc="Moviendo archivos", unit="file"):
        try:
            # Buscar imagen correspondiente
            image_found = False
            for ext in valid_exts:
                image_path = images_dir / f"{xml_path.stem}{ext}"
                if image_path.exists():
                    # Mover imagen
                    shutil.move(str(image_path), str(test_images_dir / image_path.name))
                    image_found = True
                    break
            
            if not image_found:
                print(f"⚠️  Imagen no encontrada para: {xml_path.name}")
                failed_count += 1
                continue
            
            # Mover XML
            shutil.move(str(xml_path), str(test_xml_dir / xml_path.name))
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Error moviendo {xml_path.name}: {e}")
            failed_count += 1
    
    print("\n" + "=" * 70)
    print("✅ SPLIT COMPLETADO")
    print("=" * 70)
    print(f"✅ Archivos movidos a test: {moved_count}")
    print(f"❌ Errores: {failed_count}")
    print(f"\n📁 Training set:")
    print(f"   Imágenes: {images_dir}")
    print(f"   XMLs: {xml_dir}")
    print(f"\n📁 Test set:")
    print(f"   Imágenes: {test_images_dir}")
    print(f"   XMLs: {test_xml_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Divide el dataset en conjuntos de training y testing"
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
        '--test-images-dir',
        default='dataset_testing/images',
        help='Directorio de salida para imágenes de test'
    )
    parser.add_argument(
        '--test-xml-dir',
        default='dataset_testing/xmls',
        help='Directorio de salida para XMLs de test'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Fracción del dataset para testing (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        Path(args.images_dir),
        Path(args.xml_dir),
        Path(args.test_images_dir),
        Path(args.test_xml_dir),
        args.test_split,
        args.seed
    )


if __name__ == "__main__":
    main()
