# Plan de mejoras del modelo — reducir falsos positivos ("celular → cuchillo")

## 1. Problema observado
En pruebas con video real (centro de monitoreo), el detector de armas (Faster R-CNN MobileNetV3-FPN) puede disparar **detecciones de `knife` con alta confianza** cuando aparece un **celular de costado** (objeto alargado, bordes marcados, brillo/reflectancia), generando falsos positivos.

**Objetivo principal:** bajar la tasa de falsos positivos de `knife` en el dominio objetivo (cámaras reales) sin destruir el recall en casos verdaderos.

## 2. Enfoque recomendado (prioridad alta, dataset limitado)
La estrategia más efectiva con pocos datos suele ser: **hard negatives + evaluación por casos + fine-tuning corto**.

### 2.1. Recolección de hard negatives (celulares) desde el dominio real
1) Grabar videos cortos (10–30 s) con:
- Celular en mano (de costado / de frente / con funda).
- Celular cerca de la cara (llamada), en el bolsillo, en mesa.
- Diferentes fondos e iluminaciones (interior, exterior, contraluz).

2) Extraer frames (p.ej. 1–2 fps) y **etiquetar**:
- Opción A (más simple): **no dibujar cajas de arma** (imágenes negativas).
- Opción B (mejor si se puede): etiquetar también personas (si el pipeline lo usa), pero **sin** `knife`.

3) Mezclar estos negativos con el set actual de entrenamiento.

Resultado esperado: el modelo aprende explícitamente que “celular ≠ cuchillo” en el dominio objetivo.

### 2.2. Aumentación dirigida (para el caso “objeto alargado”) 
Agregar augmentations que se parezcan a las condiciones donde falla:
- Motion blur y defocus blur moderado.
- Variaciones de iluminación (gamma/brightness/contrast más agresivas).
- Compresión JPEG fuerte (simular streaming MJPEG).
- Rotaciones pequeñas y cambios de escala.

**Nota:** priorizar augmentations que reproduzcan el “look” de cámaras reales antes que augmentations muy sintéticas.

### 2.3. Fine-tuning con control de sobreajuste
- Hacer fine-tuning desde el checkpoint actual.
- Usar early stopping (ya estaba en el pipeline) y pocas épocas.
- Monitorear métricas específicas para `knife` y, si es posible, una métrica por “escenario negativo” (celulares).

Si el dataset nuevo es chico, es preferible:
- LR bajo.
- Congelar backbone al principio (si el código ya lo soporta) y ajustar heads; luego descongelar parcialmente.

## 3. Alternativa: agregar una clase explícita `phone`
Si se cuenta con suficientes ejemplos (aunque sean pocos cientos), agregar `phone` como clase puede ayudar a que el modelo “separe” visualmente el espacio de features.

Pros:
- El modelo aprende un concepto positivo (“esto es un celular”).

Contras:
- Requiere etiquetar cajas de `phone`.
- Cambia el pipeline de entrenamiento/inferencia (nueva clase).

Recomendación práctica:
- Empezar con **hard negatives** (sección 2) y solo pasar a `phone` si el problema persiste.

## 4. Evaluación enfocada (lo que realmente importa)
Armar un set de evaluación pequeño pero representativo:
- 5–10 clips “celular” (negativos duros).
- 5–10 clips con cuchillos reales (positivos).
- 5–10 clips con objetos parecidos (control): control remoto, llaves, utensilios, herramienta fina.

Medir:
- FPR de `knife` en negativos (clave).
- Recall de `knife` en positivos.
- Curva precisión/recall variando umbral.

**Criterio de éxito mínimo sugerido:** bajar FPs de `knife` en celulares sin perder más de un pequeño porcentaje de recall en cuchillos reales.

## 5. Mitigaciones rápidas sin re-entrenar (para operar mejor ya)
Estas mitigaciones son útiles en producción/operación mientras se mejora el modelo.

### 5.1. Calibración por clase (umbral distinto para `knife` y `pistol`)
- Subir el umbral solo para `knife` (típicamente es la clase que confunde más con objetos cotidianos).

### 5.2. Consistencia temporal más exigente para `knife`
- Requerir más `frames_consecutivos` para disparar alarma de `knife`.

### 5.3. Reglas simples de plausibilidad (si aparecen muchos falsos)
Ejemplos (mantenerlas mínimas y medibles):
- Tamaño mínimo de bbox (descartar cajas muy pequeñas).
- Persistencia espacial (bbox estable en varios frames).

**Importante:** reglas excesivas pueden matar recall; usarlas solo como “parche” temporal.

## 6. Plan de implementación sugerido (1–2 semanas)
- Día 1–2: recolectar y etiquetar hard negatives (celular) + armar set de evaluación.
- Día 3–4: correr fine-tuning con negativos + augment dirigido.
- Día 5: evaluar, ajustar umbral por clase y/o frames consecutivos.
- Semana 2: si sigue el problema, considerar clase `phone` (etiquetado + retrain).

## 7. Entregables
- Carpeta con ejemplos “celular” y set de evaluación.
- Nuevo checkpoint del detector.
- Reporte corto de métricas: FPR(celular), Recall(cuchillo), y configuración recomendada de umbrales.
