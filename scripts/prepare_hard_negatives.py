#!/usr/bin/env python3
"""
Prepara los celulares como hard negatives: elimina cajas de 'knife' y 'pistol' de los XMLs.
Solo mantiene cajas de 'person' si existen (opcional).
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# Rutas
base = Path(__file__).parent.parent
images_src = base / "data/dataset_celulares/images"
xmls_src = base / "data/dataset_celulares/xmls_filtered"
output_dir = base / "data/hard_negatives_celulares"

# Crear carpetas de salida
(output_dir / "images").mkdir(parents=True, exist_ok=True)
(output_dir / "xmls").mkdir(parents=True, exist_ok=True)

# Copiar imágenes
print("📸 Copiando imágenes...")
for img in images_src.glob("*"):
    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        shutil.copy2(img, output_dir / "images" / img.name)

# Procesar XMLs
print("📝 Procesando XMLs (eliminando cajas de armas)...")
processed = 0
no_boxes = 0

for xml_path in xmls_src.glob("*.xml"):
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        
        # Eliminar todos los objetos que sean 'knife' o 'pistol'
        objects = root.findall('object')
        kept_objects = []
        
        for obj in objects:
            name_el = obj.find('name')
            if name_el is not None:
                class_name = name_el.text.strip().lower()
                # Mantener solo 'person' si existe, descartar 'knife' y 'pistol'
                if class_name not in ['knife', 'pistol']:
                    kept_objects.append(obj)
        
        # Eliminar todos los objetos originales
        for obj in objects:
            root.remove(obj)
        
        # Agregar solo los que queremos mantener (puede quedar vacío)
        for obj in kept_objects:
            root.append(obj)
        
        # Guardar XML (con o sin objetos)
        output_xml = output_dir / "xmls" / xml_path.name
        tree.write(str(output_xml), encoding='utf-8', xml_declaration=True)
        
        if len(kept_objects) == 0:
            no_boxes += 1
        
        processed += 1
        
    except Exception as e:
        print(f"❌ Error procesando {xml_path.name}: {e}")

print(f"\n✅ {processed} XMLs procesados")
print(f"📦 {no_boxes} XMLs sin cajas (hard negatives puros)")
print(f"📁 Salida: {output_dir}")
print("\n💡 Estos archivos están listos para subir a Colab como hard negatives")
