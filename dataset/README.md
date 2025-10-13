# 📁 Directorio del Dataset de Armas

Esta carpeta contiene el dataset organizado para entrenamiento del **Stage 2** (detección de armas).

## 🏗️ Estructura Recomendada

```
dataset/
├── original/                    # Imágenes originales
│   ├── knife/                  # Cuchillos originales
│   │   ├── knife_001.jpg
│   │   ├── knife_002.jpg
│   │   └── ...
│   └── pistol/                 # Pistolas originales
│       ├── pistol_001.jpg
│       ├── pistol_002.jpg
│       └── ...
│
└── augmented/                  # Dataset aumentado (generado automáticamente)
    ├── knife/                  # Cuchillos aumentados (objetivo: 1500+)
    │   ├── knife_original_0001.jpg
    │   ├── knife_aug_0001_001_flip_horizontal.jpg
    │   ├── knife_aug_0001_002_rotate_90.jpg
    │   └── ...
    └── pistol/                 # Pistolas aumentadas (objetivo: 1500+)
        ├── pistol_original_0001.jpg
        ├── pistol_aug_0001_001_flip_vertical.jpg
        └── ...
```

## 🚀 Uso del Sistema de Data Augmentation

### **Opción 1: Organización por Subdirectorios (Recomendada)**
```bash
# 1. Crear subdirectorios por clase
mkdir -p dataset/original/knife
mkdir -p dataset/original/pistol

# 2. Copiar imágenes a sus respectivas carpetas
cp cuchillo_*.jpg dataset/original/knife/
cp pistola_*.jpg dataset/original/pistol/

# 3. Ejecutar augmentation
python data_augmentation.py --input dataset/original --target 1500
```

### **Opción 2: Detección Automática por Nombre**
```bash
# 1. Copiar todas las imágenes a original/
cp *.jpg dataset/original/

# 2. El sistema detectará automáticamente por nombres:
#    - knife, cuchillo, blade → clase 'knife'
#    - gun, pistol, pistola → clase 'pistol'

# 3. Ejecutar augmentation
python data_augmentation.py --input dataset/original --target 2000
```

## ⚙️ Parámetros de Configuración

### **Básico:**
```bash
python data_augmentation.py
# Usa defaults: 1500 imágenes por clase
```

### **Personalizado:**
```bash
python data_augmentation.py \
  --input mis_imagenes/ \
  --output dataset_aumentado/ \
  --target 2000
```

### **Para llegar a 2000 imágenes:**
```bash
python data_augmentation.py --target 2000
```

## 🔄 Transformaciones Aplicadas

### **Transformaciones Principales:**
1. **Flip Horizontal** (15%) - Muy efectivo para armas
2. **Flip Vertical** (10%) - Diferentes orientaciones
3. **Rotación 90°** (15%) - Cambios de orientación preservando calidad
4. **Rotación 180°** (10%) - Inversión completa
5. **Rotación 270°** (10%) - Orientación adicional

### **Transformaciones Finas:**
6. **Rotación ±15°** (8% c/u) - Variaciones sutiles
7. **Rotación ±30°** (6% c/u) - Variaciones moderadas  
8. **Rotación ±45°** (4% c/u) - Variaciones amplias
9. **Escalado 0.8x** (5%) - Zoom out con padding blanco
10. **Escalado 1.2x** (5%) - Zoom in para mayor detalle

### **Transformaciones Combinadas:**
11. **Flip + Rotación** (3%) - Transformaciones compuestas
12. **Brillo/Contraste** (3%) - Variaciones de iluminación

## 📊 Resultados Esperados

### **Entrada Típica:**
- 📸 **Cuchillos**: 50 imágenes originales
- 📸 **Pistolas**: 45 imágenes originales  
- 🎯 **Total original**: 95 imágenes

### **Salida con target=1500:**
- 🔪 **Cuchillos**: 1500 imágenes (factor 30x)
- 🔫 **Pistolas**: 1500 imágenes (factor 33x)
- 🎯 **Total final**: 3000 imágenes

### **Distribución de Transformaciones:**
```
flip_horizontal    : ~450 imágenes (15%)
rotate_90         : ~450 imágenes (15%)  
flip_vertical     : ~300 imágenes (10%)
rotate_180        : ~300 imágenes (10%)
rotate_fine       : ~600 imágenes (20%)
scaling           : ~300 imágenes (10%)
combined          : ~180 imágenes (6%)
original_copies   : ~95 imágenes (copias)
```

## 🎯 Optimizaciones para Armas

### **Fondos Blancos/Limpios:**
- ✅ **Rotaciones** preservan el fondo blanco
- ✅ **Padding blanco** en escalado
- ✅ **Sin transformaciones** que dañen el contraste objeto-fondo

### **Preservación de Características:**
- ✅ **Formas distintivas** de armas mantenidas
- ✅ **Proporciones** preservadas en escalado
- ✅ **Calidad** mantenida con interpolación cúbica

### **Variabilidad Realista:**
- ✅ **Orientaciones** que pueden ocurrir en detección real
- ✅ **Escalas** apropiadas para diferentes distancias
- ✅ **Transformaciones** ponderadas por utilidad

## 📈 Monitoreo del Proceso

El sistema genera reportes en tiempo real:
```
🔄 Procesando clase: knife
📁 Imágenes originales encontradas: 50
🎯 Objetivo: 1500 imágenes
➕ Necesarias: 1450 adicionales
🔢 Augmentaciones por imagen: 29

  📸 Procesando knife_001.jpg (1/50)
    ➡️ Generadas: 50
    ➡️ Generadas: 100
    ...

✅ Clase knife completada:
   📊 Originales: 50
   ➕ Generadas: 1450
   🎯 Total final: 1500
```

## ⚠️ Consideraciones Importantes

### **Calidad del Dataset:**
- 🔍 **Revisar** imágenes generadas para detectar artefactos
- 🎯 **Balancear** clases para evitar bias
- 🧹 **Limpiar** manualmente si es necesario

### **Entrenamiento Posterior:**
- 📊 **Split**: 70% train, 15% validation, 15% test
- 🔄 **Shuffle** antes del entrenamiento
- ⚖️ **Balanceo** entre clases original/aumentada

### **Storage:**
- 💾 **Espacio requerido**: ~500MB para 3000 imágenes HD
- 📁 **Organización** automática por clases
- 🏷️ **Nomenclatura** sistemática para trazabilidad

---

**💡 Tip**: Comienza con un target pequeño (ej: 200) para probar el sistema antes de generar el dataset completo.