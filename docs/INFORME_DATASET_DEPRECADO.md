# 📊 Informe de Dataset - Procesamiento de Imágenes (DEPRECADO)

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

**Total de personas extraídas**: 270

**Tamaño promedio original**: 147x66 píxeles

**Tamaño mínimo**: 55x36 píxeles

**Tamaño máximo**: 253x117 píxeles


### 🎨 Pipeline de Mejora de Calidad

**Estado**: ✅ Pipeline aplicado correctamente


Técnicas de mejora aplicadas:
1. ✅ **Interpolación Spline Cúbica** - Redimensionamiento a mínimo 200x100 px
2. ✅ **Mejora de Contraste** - CLAHE adaptativo
3. ✅ **Mejora de Brillo** - Mejora el brillo de la imagen aumentando el canal V en HSV

**Objetivo**: Mejorar la calidad de las imágenes de personas para facilitar
la detección de armas en el Stage 2 del pipeline.