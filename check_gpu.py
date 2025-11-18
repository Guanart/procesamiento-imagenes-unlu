#!/usr/bin/env python3
"""
Script de verificación GPU AMD ROCm (Radeon 780M iGPU)
Basado en recomendación de la profesora para WSL2 + AMD
"""
import os
import torch

# --- Configuración optimizada para AMD ROCm ---
# Reduce fragmentación de memoria
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("🔍 VERIFICACIÓN GPU AMD ROCm")
print("=" * 60)
print(f"Versión PyTorch: {torch.__version__}")
print(f"¿ROCm/GPU disponible?: {torch.cuda.is_available()}")
print(f"Dispositivo seleccionado: {DEVICE}")

# Verificar CPU
import platform
print(f"\n💻 Información del Sistema:")
print(f"   Procesador: {platform.processor()}")
print(f"   Threads disponibles: {torch.get_num_threads()}")

if torch.cuda.is_available():
    print(f"\n🎮 GPU Detectada:")
    print(f"   Nombre: {torch.cuda.get_device_name(0)}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Memoria total: {total_mem_gb:.2f} GB")
    
    # Test simple de operación en GPU
    print("\n🧪 Probando operación en GPU...")
    try:
        x = torch.randn(1000, 1000, device=DEVICE)
        y = torch.randn(1000, 1000, device=DEVICE)
        z = torch.matmul(x, y)
        print("✅ Multiplicación de matrices 1000x1000 exitosa")
        print(f"   Resultado shape: {z.shape}, device: {z.device}")
    except Exception as e:
        print(f"❌ Error en operación GPU: {e}")
else:
    print("\n⚠️  GPU no disponible - Ejecutando en CPU")
    print("\n📋 Estado de drivers ROCm:")
    import os
    has_kfd = os.path.exists("/dev/kfd")
    has_dri = os.path.exists("/dev/dri")
    print(f"   /dev/kfd (ROCm kernel): {'✅ Presente' if has_kfd else '❌ Ausente'}")
    print(f"   /dev/dri (GPU access): {'✅ Presente' if has_dri else '❌ Ausente'}")
    
    if not has_kfd and not has_dri:
        print("\n💡 Nota: AMD Radeon 780M (iGPU) detectada en CPU")
        print("   ROCm en WSL2 tiene soporte limitado para iGPUs integradas.")
        print("   El entrenamiento continuará usando CPU (optimizado).")
    
    # Test de CPU
    print("\n🧪 Probando operación en CPU...")
    try:
        import time
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        start = time.time()
        z = torch.matmul(x, y)
        elapsed = time.time() - start
        print(f"✅ Multiplicación de matrices 1000x1000 exitosa")
        print(f"   Tiempo: {elapsed*1000:.2f}ms")
        print(f"   Threads usados: {torch.get_num_threads()}")
    except Exception as e:
        print(f"❌ Error en operación CPU: {e}")

print("=" * 60)
