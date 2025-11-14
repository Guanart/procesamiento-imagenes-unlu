#!/usr/bin/env python3
"""
Script de verificación GPU AMD ROCm (Radeon 780M iGPU)
Basado en recomendación de la profesora para WSL2 + AMD
"""
import os
import torch

# --- Configuración optimizada para AMD ROCm ---
# Reduce fragmentación de memoria
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("🔍 VERIFICACIÓN GPU AMD ROCm")
print("=" * 60)
print(f"¿ROCm/GPU disponible?: {torch.cuda.is_available()}")
print(f"Dispositivo seleccionado: {DEVICE}")

if torch.cuda.is_available():
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Memoria total GPU: {total_mem_gb:.2f} GB")
    print(f"Versión PyTorch: {torch.__version__}")
    
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
    print("⚠️  GPU no disponible, usando CPU")
    print("\nPara habilitar ROCm en WSL2:")
    print("1. Instalar PyTorch ROCm:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
    print("2. Verificar drivers AMD en WSL2")
    print("3. Verificar variable PYTORCH_HIP_ALLOC_CONF")

print("=" * 60)
