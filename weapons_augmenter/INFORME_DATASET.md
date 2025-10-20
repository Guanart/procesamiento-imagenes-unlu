# 📊 Informe de Dataset - Procesamiento de Imágenes

**Fecha de generación**: 2025-10-20 20:58:42

---

## 🔫 Dataset de Armas (Cuchillos y Pistolas)

### 📈 Balanceo de Clases

**Estado**: ⚠️ Desbalanceado

**Total de imágenes**: 5680

- **knife**: 2540 imágenes (44.7%)
- **pistol**: 3140 imágenes (55.3%)

⚠️ Diferencia entre clases: 600 imágenes

💡 **Recomendación**: Aplicar data augmentation para balancear

### 📐 Normalización de Tamaño


#### Knife

- **Cantidad**: 2540 imágenes
- **Tamaño promedio**: 157x183 píxeles
- **Tamaño mínimo**: 120x120 píxeles
- **Tamaño máximo**: 3000x3000 píxeles
- **Data Augmentation**: ✅ Aplicada

#### Pistol

- **Cantidad**: 3140 imágenes
- **Tamaño promedio**: 139x162 píxeles
- **Tamaño mínimo**: 120x120 píxeles
- **Tamaño máximo**: 300x300 píxeles
- **Data Augmentation**: ✅ Aplicada

### 🔄 Transformaciones Aplicadas

Las siguientes transformaciones básicas han sido aplicadas:
1. ✅ **Flip horizontal** - Volteo espejo
2. ✅ **Flip vertical** - Volteo vertical
3. ✅ **Rotación 90°** - Rotación en sentido horario

💡 Estas transformaciones cuadruplican el dataset original.

---

## 👥 Dataset de Personas (Stage 2)

⚠️ No se encontraron datos del dataset de personas.


💡 **Recomendación**: Ejecutar `video_processor.py` para extraer personas de videos.

---

## 📋 Conclusiones y Recomendaciones

- ⚠️ El dataset de armas está **desbalanceado**, considerar más augmentation

### 🎯 Próximos Pasos

1. Verificar el balanceo de clases en el dataset de armas
2. Asegurar que todas las imágenes de personas estén mejoradas
3. Proceder con el entrenamiento del modelo Stage 2
4. Aplicar fine-tuning de YOLOv8 para detección de armas en personas

---

*Informe generado automáticamente por `generate_report.py`*