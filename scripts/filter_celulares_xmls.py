#!/usr/bin/env python3
"""
Filtra XMLs para que solo queden los que corresponden a las imágenes de celulares.
"""
from pathlib import Path
import shutil

# Rutas
base = Path(__file__).parent.parent
images_dir = base / "data/dataset_celulares/images"
xmls_source = base / "data/dataset_celulares/xmls/xmls"
xmls_filtered = base / "data/dataset_celulares/xmls_filtered"

# Crear carpeta de salida
xmls_filtered.mkdir(exist_ok=True)

# Leer imágenes
images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
print(f"📸 {len(images)} imágenes encontradas")

# Filtrar XMLs correspondientes
copied = 0
missing = []

for img in images:
    stem = img.stem  # sin extensión
    xml_path = xmls_source / f"{stem}.xml"
    
    if xml_path.exists():
        shutil.copy2(xml_path, xmls_filtered / xml_path.name)
        copied += 1
    else:
        missing.append(stem)

print(f"✅ {copied} XMLs copiados a {xmls_filtered}")

if missing:
    print(f"⚠️  {len(missing)} XMLs no encontrados:")
    for m in missing:
        print(f"   - {m}.xml")
else:
    print("✨ Todos los XMLs encontrados")
