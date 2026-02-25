#!/usr/bin/env python3
"""
Empaqueta los hard negatives en un ZIP para subir fácilmente a Colab.
"""
import zipfile
from pathlib import Path

base = Path(__file__).parent.parent
source = base / "data/hard_negatives_celulares"
output_zip = base / "data/hard_negatives_celulares.zip"

print(f"📦 Empaquetando hard negatives...")

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Agregar imágenes
    images_dir = source / "images"
    for img in images_dir.glob("*"):
        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            arcname = f"hard_negatives/images/{img.name}"
            zf.write(img, arcname)
    
    # Agregar XMLs
    xmls_dir = source / "xmls"
    for xml in xmls_dir.glob("*.xml"):
        arcname = f"hard_negatives/xmls/{xml.name}"
        zf.write(xml, arcname)

size_mb = output_zip.stat().st_size / (1024 * 1024)
print(f"✅ ZIP creado: {output_zip}")
print(f"📊 Tamaño: {size_mb:.2f} MB")
print(f"\n💡 Subí este archivo a Drive o directamente al Colab")
