# 📘 Guía de Entrenamiento - Faster R-CNN Weapon Detection

## 🚀 Entrenamiento Completo (1000 épocas con checkpoints)

### Comando básico:
```bash
python train_fasterrcnn_light.py \
  --images-dir dataset_augmented/images \
  --xml-dir dataset_augmented/xmls \
  --output-dir results_full \
  --epochs 1000 \
  --batch-size 8 \
  --lr 1e-4 \
  --enhance \
  --amp \
  --save-every 15 \
  --patience 5
```

## 📋 Explicación de Parámetros

| Parámetro | Descripción | Valor Recomendado |
|-----------|-------------|-------------------|
| `--epochs` | Número total de épocas | `1000` |
| `--save-every` | Guardar checkpoint cada N épocas | `15` |
| `--patience` | Early stopping (sin mejora en val_loss) | `5` |
| `--enhance` | Activar CLAHE + Brightness | `True` |
| `--amp` | Automatic Mixed Precision (velocidad) | `True` |
| `--batch-size` | Imágenes por batch | `8` (ajustar según VRAM) |

## 💾 Sistema de Checkpoints

### Archivos generados:

```
results_full/
├── best_model.pth              # Mejor modelo por mAP
├── checkpoint_epoch_15.pth     # Checkpoint cada 15 épocas
├── checkpoint_epoch_30.pth
├── checkpoint_epoch_45.pth
├── ...
├── training_log.json           # Historial completo
├── training_history.png        # Gráficos de métricas
└── classes.json                # Mapa de clases
```

### Contenido de un checkpoint:
- `model_state_dict`: Pesos del modelo
- `optimizer_state_dict`: Estado del optimizador
- `epoch`: Número de época
- `train_loss`: Loss de entrenamiento
- `val_loss`: Loss de validación
- `map`: Mean Average Precision
- `best_map`: Mejor mAP hasta el momento
- `history`: Historial completo de entrenamiento

## 🔄 Reanudar Entrenamiento

Si el entrenamiento se interrumpe (corte de luz, OOM, etc.), puedes reanudarlo:

```bash
# Reanudar desde el último checkpoint
python train_fasterrcnn_light.py \
  --images-dir dataset_augmented/images \
  --xml-dir dataset_augmented/xmls \
  --output-dir results_full \
  --epochs 1000 \
  --batch-size 8 \
  --enhance \
  --amp \
  --save-every 15 \
  --patience 5 \
  --resume results_full/checkpoint_epoch_45.pth
```

**⚠️ Importante:** Usa los **mismos parámetros** que el entrenamiento original (batch-size, lr, etc.)

## ⏸️ Early Stopping

El entrenamiento se detendrá automáticamente si:

- El **validation loss** no mejora durante `--patience` épocas (default: 5)
- Se alcanza el límite de RAM configurado
- Se completan todas las épocas especificadas

### Ejemplo de salida:

```
--- Epoch 45/1000 (RAM: 12.34GB) ---
🏋️ Training: 100%|████████| 625/625 [02:15<00:00, 4.62batch/s, loss=0.2341]
🔍 Validating: 100%|██████████| 110/110 [00:18<00:00, 6.11batch/s]
✅ Epoch 45 | Train Loss: 0.2341 | Val Loss: 0.5678 | mAP: 0.7234 | Time: 153.2s
⚠️  Val Loss no mejoró (3/5)
📦 Checkpoint guardado: checkpoint_epoch_45.pth
```

## 📊 Monitoreo del Progreso

### Verificar checkpoints guardados:
```bash
ls -lh results_full/checkpoint_*.pth
```

### Cargar y analizar un checkpoint en Python:
```python
import torch

checkpoint = torch.load('results_full/checkpoint_epoch_45.pth')
print(f"Época: {checkpoint['epoch']}")
print(f"Train Loss: {checkpoint['train_loss']:.4f}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"mAP: {checkpoint['map']:.4f}")
print(f"Mejor mAP: {checkpoint['best_map']:.4f}")
```

## 🎯 Estrategias de Entrenamiento

### 1. Exploración rápida (encontrar hiperparámetros)
```bash
python train_fasterrcnn_light.py \
  --epochs 50 \
  --batch-size 16 \
  --save-every 10 \
  --patience 5
```

### 2. Entrenamiento largo con seguridad
```bash
python train_fasterrcnn_light.py \
  --epochs 1000 \
  --batch-size 8 \
  --save-every 15 \
  --patience 10 \
  --enhance \
  --amp
```

### 3. Fine-tuning desde modelo existente
```bash
python train_fasterrcnn_light.py \
  --epochs 200 \
  --lr 1e-5 \
  --save-every 10 \
  --resume results_full/best_model.pth
```

## 🛡️ Protección contra Fallos

### El sistema de checkpoints protege contra:

✅ **Cortes de luz**: Reanudar desde último checkpoint  
✅ **OOM (Out of Memory)**: Reducir batch-size y reanudar  
✅ **Errores del kernel**: Reiniciar y continuar  
✅ **Desconexión de Colab**: Checkpoints guardados en Drive  

### Buenas prácticas:

1. **Usar Google Drive** en Colab para persistir checkpoints
2. **Verificar espacio** antes de entrenar (checkpoints ocupan ~100MB c/u)
3. **Guardar cada 10-20 épocas** (balance entre seguridad y espacio)
4. **Monitorear val_loss** para detectar overfitting temprano

## 📈 Interpretación de Resultados

### Señales de buen entrenamiento:
- ✅ Train Loss desciende suavemente
- ✅ Val Loss desciende o se mantiene estable
- ✅ mAP aumenta progresivamente
- ✅ Gap pequeño entre train_loss y val_loss

### Señales de problemas:
- ❌ Val Loss aumenta mientras train_loss baja (overfitting)
- ❌ Ambos loss se estancan temprano (learning rate muy bajo)
- ❌ Loss explota (learning rate muy alto)
- ❌ mAP no supera 0.3 después de 50 épocas (problema en datos)

## 🔧 Troubleshooting

### Problema: OOM (Out of Memory)
**Solución:**
```bash
# Reducir batch-size
--batch-size 4

# O desactivar AMP si causa problemas
# (quitar --amp)
```

### Problema: Entrenamiento muy lento
**Solución:**
```bash
# Activar AMP
--amp

# Reducir resolución (editar RESIZE_TO en código)
RESIZE_TO = (320, 320)  # más rápido
RESIZE_TO = (640, 640)  # mejor calidad
```

### Problema: Overfitting temprano
**Solución:**
```bash
# Aumentar regularización (editar código)
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

# O usar más augmentation
python augment_dataset.py --num-augmentations 5
```

## 📞 Comandos Útiles

### Ver historial de entrenamiento:
```python
import json
with open('results_full/training_log.json') as f:
    history = json.load(f)
    
for epoch in history[-10:]:  # Últimas 10 épocas
    print(f"Epoch {epoch['epoch']}: mAP={epoch['map']:.4f}, Val Loss={epoch['val_loss']:.4f}")
```

### Eliminar checkpoints antiguos (ahorrar espacio):
```bash
# Mantener solo cada 3er checkpoint
rm results_full/checkpoint_epoch_{15,30,60,75,90}.pth
# (mantener 45, 105, 150, etc.)
```

---

**¿Dudas?** Revisa `training_history.png` para ver gráficos de progreso.
