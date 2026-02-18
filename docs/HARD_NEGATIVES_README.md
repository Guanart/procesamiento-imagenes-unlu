# Fine-tuning con Hard Negatives (Celulares)

## 🎯 Objetivo
Reducir falsos positivos de `knife` cuando el modelo confunde **celulares de costado** con cuchillos.

## 📦 Archivos Preparados

### Datos locales
- **Imágenes:** `data/hard_negatives_celulares/images/` (28 imágenes)
- **XMLs:** `data/hard_negatives_celulares/xmls/` (28 anotaciones SIN armas)
- **ZIP:** `data/hard_negatives_celulares.zip` (11.71 MB, listo para subir)

### Código
- **Script de filtrado:** `scripts/filter_celulares_xmls.py`
- **Script de preparación:** `scripts/prepare_hard_negatives.py`
- **Script de empaquetado:** `scripts/zip_hard_negatives.py`
- **Notebook simple:** `notebooks/finetuning_simple.ipynb` ⭐ **RECOMENDADO**
- **Notebook completo:** `notebooks/finetuning_hard_negatives.ipynb`

### Modificaciones de código
- ✅ `train_fasterrcnn_light.py` ya modificado para soportar imágenes sin cajas (hard negatives).

## 🚀 Ejecutar desde VSCode con Colab

### Opción 1: Subir ZIP a Drive (Recomendada)

1. **Subir ZIP a Drive:**
   ```bash
   # Desde tu máquina local, subí el archivo a Drive:
   # data/hard_negatives_celulares.zip
   # → Drive/MyDrive/procesamiento-imagenes/
   ```

2. **Abrir notebook en VSCode:**
   - Abrir `notebooks/finetuning_simple.ipynb` en VSCode
   - Asegurate de tener la extensión de Colab instalada
   - Conectar a un runtime de Colab con GPU (T4)

3. **Ejecutar celdas en orden:**
   - El notebook copia todo desde Drive automáticamente
   - Integra los 28 celulares al training
   - Ejecuta fine-tuning con hiperparámetros optimizados

### Opción 2: Upload directo desde VSCode

1. **Abrir:** `notebooks/finetuning_hard_negatives.ipynb`
2. **Ejecutar hasta la celda de upload**
3. **Subir manualmente:**
   - Desde la carpeta: `data/hard_negatives_celulares/images/` (28 archivos)
   - Desde la carpeta: `data/hard_negatives_celulares/xmls/` (28 archivos)

> ⚠️ Nota: El widget de upload puede ser lento. Recomendamos **Opción 1**.

## 📊 Verificar Mejora

### Métricas a monitorear

**Objetivo principal:**
- ✅ FP de `knife` debe bajar
- ✅ Precision de `knife` debe subir

**Verificar que no empeore:**
- ⚠️ Recall de `knife` no debe bajar más de 2-3%
- ⚠️ mAP general debe mantenerse similar

### Comparación

```bash
# Modelo anterior (antes de hard negatives)
FP knife: X
Recall knife: Y%
Precision knife: Z%

# Modelo nuevo (con hard negatives)
FP knife: X - ΔX  # Debe bajar
Recall knife: Y%  # Debe mantenerse
Precision knife: Z + ΔZ%  # Debe subir
```

## 🔬 Próximos Pasos

Si los FPs siguen siendo altos:

1. **Más hard negatives:**
   - Grabar más videos con celular de costado
   - Diferentes iluminaciones y fondos
   - Agregar al dataset con el mismo proceso

2. **Augmentations dirigidas:**
   - Motion blur (simular video)
   - Compresión JPEG fuerte (simular streaming)
   - Variaciones de brillo/contraste

3. **Clase explícita `phone`:**
   - Si el problema persiste, considerar agregar `phone` como tercera clase
   - Requiere re-etiquetar con cajas de celular

4. **Calibración de umbrales:**
   - Subir umbral solo para `knife` en la app de monitoreo
   - Requerir más frames consecutivos para `knife`

## 📝 Comandos Rápidos

```bash
# Filtrar XMLs (si agregás más imágenes)
python3 scripts/filter_celulares_xmls.py

# Preparar hard negatives (eliminar cajas de armas)
python3 scripts/prepare_hard_negatives.py

# Empaquetar en ZIP
python3 scripts/zip_hard_negatives.py
```

## ✅ Checklist

Antes de ejecutar:
- [ ] ZIP subido a Drive (`hard_negatives_celulares.zip`)
- [ ] Checkpoint del modelo actual disponible en Drive
- [ ] Runtime de Colab con GPU T4 conectado
- [ ] Extensión de Colab instalada en VSCode

Durante el entrenamiento:
- [ ] Verificar que training tiene 28 imágenes más
- [ ] Monitorear loss (debe bajar gradualmente)
- [ ] Early stopping activado (patience=5)

Después del entrenamiento:
- [ ] Comparar FP de knife (antes vs después)
- [ ] Verificar recall de knife (no debe bajar mucho)
- [ ] Probar modelo en video con celular de costado
- [ ] Copiar resultados a Drive

## 🆘 Troubleshooting

**Error: "imagen sin XML"**
- Ejecutar `filter_celulares_xmls.py` de nuevo

**Error: "boxes vacías"**
- ✅ Es esperado. Los hard negatives NO tienen cajas de armas
- El código ya está preparado para esto

**Loss no baja:**
- Verificar que `--resume` apunta al checkpoint correcto
- Bajar LR a `5e-6` si 1e-5 es muy alto

**Recall de knife baja mucho:**
- Parar el entrenamiento antes (menos épocas)
- Subir LR a `3e-5` para convergencia más rápida
- Reducir cantidad de hard negatives (usar solo 15 en vez de 28)

## 📧 Contacto

Cualquier duda, revisar:
- `MEJORAS_MODELO.md` (plan completo de mejora)
- `INFORME_FINAL.md` (contexto del proyecto)
