# Weapons Augmenter

Aumentación simple de dataset de armas mediante transformaciones geométricas.

## Descripción

Aplica 3 transformaciones básicas a cada imagen:
- Flip horizontal
- Flip vertical  
- Rotación 90°

**Resultado:** Dataset cuadruplicado (original + 3 variaciones)

## Uso

```bash
python simple_augmenter.py --input dataset/original --output dataset/augmented
```

## Estructura

```
weapons_augmenter/
├── simple_augmenter.py     # Script principal
└── dataset/
    ├── original/           # Imágenes originales (785 pistols + 635 knives)
    │   ├── pistol/
    │   └── knife/
    └── augmented/          # Imágenes aumentadas (3,140 pistols + 2,540 knives)
        ├── pistol/
        └── knife/
```

## Resultados

- **Original:** 1,420 imágenes
- **Aumentado:** 5,680 imágenes (4x)
- **Balance:** Mantiene proporción 55-45 (pistols-knives)
