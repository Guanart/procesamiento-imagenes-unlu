#!/usr/bin/env python3
"""
Pipeline Completo de Entrenamiento
Automatiza el flujo completo: Split → Augment → Train → Test

Ejecuta todos los scripts necesarios en secuencia:
1. split_dataset.py: Separa 10% para testing
2. augment_dataset.py: Aumenta el conjunto de training
3. train_fasterrcnn_light.py: Entrena con checkpoints
4. test_light_model.py: Evalúa métricas finales en test set

Autor: Procesamiento de Imágenes - UNLU
Fecha: Noviembre 2025
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import time


class TrainingPipeline:
    """Orquestador del pipeline completo de entrenamiento."""
    
    def __init__(self, config):
        self.config = config
        self.start_time = None
        self.results = {
            "pipeline_start": None,
            "pipeline_end": None,
            "stages": {},
            "success": False
        }
    
    def run_command(self, stage_name: str, command: list) -> bool:
        """
        Ejecuta un comando y registra el resultado.
        
        Args:
            stage_name: Nombre de la etapa
            command: Lista con comando y argumentos
            
        Returns:
            True si exitoso, False si falló
        """
        print("\n" + "=" * 70)
        print(f"🚀 ETAPA: {stage_name}")
        print("=" * 70)
        print(f"📝 Comando: {' '.join(command)}")
        print()
        
        stage_start = time.time()
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,  # Mostrar output en tiempo real
                text=True
            )
            
            stage_end = time.time()
            duration = stage_end - stage_start
            
            self.results["stages"][stage_name] = {
                "success": True,
                "duration_sec": duration,
                "command": " ".join(command)
            }
            
            print()
            print(f"✅ {stage_name} completado en {duration:.1f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            stage_end = time.time()
            duration = stage_end - stage_start
            
            self.results["stages"][stage_name] = {
                "success": False,
                "duration_sec": duration,
                "error": str(e),
                "command": " ".join(command)
            }
            
            print()
            print(f"❌ {stage_name} falló después de {duration:.1f}s")
            print(f"Error: {e}")
            return False
    
    def stage_1_split(self) -> bool:
        """Etapa 1: Separar dataset en training y testing."""
        command = [
            "python3", "split_dataset.py",
            "--images-dir", self.config["dataset_images"],
            "--xml-dir", self.config["dataset_xmls"],
            "--test-images-dir", self.config["test_images"],
            "--test-xml-dir", self.config["test_xmls"],
            "--test-split", str(self.config["test_split"]),
            "--seed", str(self.config["seed"])
        ]
        return self.run_command("1. Split Dataset", command)
    
    def stage_2_augment(self) -> bool:
        """Etapa 2: Aumentar el conjunto de training."""
        command = [
            "python3", "augment_dataset.py",
            "--images-dir", self.config["dataset_images"],
            "--xml-dir", self.config["dataset_xmls"],
            "--output-images-dir", self.config["augmented_images"],
            "--output-xml-dir", self.config["augmented_xmls"],
            "--num-augmentations", str(self.config["num_augmentations"])
        ]
        
        if self.config["copy_originals"]:
            command.append("--copy-originals")
        
        return self.run_command("2. Data Augmentation", command)
    
    def stage_3_train(self) -> bool:
        """Etapa 3: Entrenar modelo con checkpoints."""
        command = [
            "python3", "train_fasterrcnn_light.py",
            "--images-dir", self.config["augmented_images"],
            "--xml-dir", self.config["augmented_xmls"],
            "--output-dir", self.config["output_dir"],
            "--epochs", str(self.config["epochs"]),
            "--batch-size", str(self.config["batch_size"]),
            "--lr", str(self.config["learning_rate"]),
            "--save-every", str(self.config["save_every"]),
            "--patience", str(self.config["patience"])
        ]
        
        if self.config["enhance"]:
            command.append("--enhance")
        
        if self.config["amp"]:
            command.append("--amp")
        
        if self.config["resume"]:
            command.extend(["--resume", self.config["resume"]])
        
        return self.run_command("3. Train Model", command)
    
    def stage_4_test(self) -> bool:
        """Etapa 4: Evaluar modelo en test set."""
        # Buscar el mejor modelo
        output_dir = Path(self.config["output_dir"])
        best_model = output_dir / "best_model.pth"
        
        if not best_model.exists():
            # Buscar checkpoint final
            checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoints:
                best_model = checkpoints[-1]
            else:
                print("❌ No se encontró ningún modelo entrenado")
                return False
        
        command = [
            "python3", "test_light_model.py",
            "--model", str(best_model),
            "--test-images-dir", self.config["test_images"],
            "--test-xml-dir", self.config["test_xmls"],
            "--output", self.config["test_output"],
            "--confidence", str(self.config["confidence"])
        ]
        
        if self.config["no_save_test_images"]:
            command.append("--no-save-images")
        
        return self.run_command("4. Test Model", command)
    
    def run(self) -> bool:
        """Ejecuta el pipeline completo."""
        self.start_time = time.time()
        self.results["pipeline_start"] = datetime.now().isoformat()
        
        print("\n" + "=" * 70)
        print("🎯 PIPELINE DE ENTRENAMIENTO COMPLETO")
        print("=" * 70)
        print(f"📅 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Dataset original: {self.config['dataset_images']}")
        print(f"📂 Salida: {self.config['output_dir']}")
        print("=" * 70)
        
        # Verificar que estamos en el directorio correcto
        if not Path("split_dataset.py").exists():
            print("❌ Error: No se encontró split_dataset.py")
            print("   Ejecuta este script desde weapons_detector2/")
            return False
        
        # Ejecutar etapas
        stages = [
            ("split", self.stage_1_split),
            ("augment", self.stage_2_augment),
            ("train", self.stage_3_train),
            ("test", self.stage_4_test)
        ]
        
        for stage_name, stage_func in stages:
            if stage_name in self.config["skip_stages"]:
                print(f"\n⏭️  Saltando etapa: {stage_name}")
                continue
            
            success = stage_func()
            if not success and not self.config["continue_on_error"]:
                print(f"\n❌ Pipeline detenido en etapa: {stage_name}")
                self.save_results(success=False)
                return False
        
        # Pipeline completado
        self.save_results(success=True)
        return True
    
    def save_results(self, success: bool):
        """Guarda resumen del pipeline."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        self.results["pipeline_end"] = datetime.now().isoformat()
        self.results["total_duration_sec"] = total_duration
        self.results["success"] = success
        
        # Guardar JSON
        results_file = Path(self.config["output_dir"]) / "pipeline_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Mostrar resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN DEL PIPELINE")
        print("=" * 70)
        print(f"⏱️  Duración total: {total_duration / 60:.1f} minutos")
        print(f"📝 Resultado: {'✅ EXITOSO' if success else '❌ FALLÓ'}")
        print()
        print("📋 Etapas ejecutadas:")
        for stage_name, stage_data in self.results["stages"].items():
            status = "✅" if stage_data["success"] else "❌"
            duration = stage_data["duration_sec"]
            print(f"   {status} {stage_name}: {duration:.1f}s")
        
        print(f"\n💾 Resultados guardados en: {results_file}")
        
        if success:
            print("\n🎉 Pipeline completado exitosamente!")
            print(f"📁 Modelo entrenado: {self.config['output_dir']}/best_model.pth")
            print(f"📊 Métricas de test: {self.config['test_output']}/test_metrics.json")
        
        print("=" * 70)


def get_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de entrenamiento: Split → Augment → Train → Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Pipeline completo con configuración por defecto
  python pipeline_entrenamiento.py

  # Entrenamiento largo con más augmentación
  python pipeline_entrenamiento.py \\
    --epochs 1000 \\
    --num-augmentations 3 \\
    --save-every 20

  # Reanudar entrenamiento después de interrupción
  python pipeline_entrenamiento.py \\
    --skip-stages split augment \\
    --resume results_full/checkpoint_epoch_45.pth

  # Solo testing (modelo ya entrenado)
  python pipeline_entrenamiento.py \\
    --skip-stages split augment train
"""
    )
    
    # Directorios
    parser.add_argument("--dataset-images", default="dataset/images",
                        help="Directorio con imágenes originales")
    parser.add_argument("--dataset-xmls", default="dataset/xmls",
                        help="Directorio con XMLs originales")
    parser.add_argument("--test-images", default="dataset_testing/images",
                        help="Directorio para imágenes de test")
    parser.add_argument("--test-xmls", default="dataset_testing/xmls",
                        help="Directorio para XMLs de test")
    parser.add_argument("--augmented-images", default="dataset_augmented/images",
                        help="Directorio para imágenes aumentadas")
    parser.add_argument("--augmented-xmls", default="dataset_augmented/xmls",
                        help="Directorio para XMLs aumentados")
    parser.add_argument("--output-dir", default="results_full",
                        help="Directorio de salida para modelo")
    parser.add_argument("--test-output", default="test_results",
                        help="Directorio para resultados de test")
    
    # Parámetros de split
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fracción para test (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    
    # Parámetros de augmentation
    parser.add_argument("--num-augmentations", type=int, default=2,
                        help="Versiones aumentadas por imagen")
    parser.add_argument("--copy-originals", action="store_true",
                        help="Copiar originales al dataset aumentado")
    
    # Parámetros de entrenamiento
    parser.add_argument("--epochs", type=int, default=100,
                        help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Tamaño del batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--save-every", type=int, default=15,
                        help="Guardar checkpoint cada N épocas")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--enhance", action="store_true",
                        help="Activar mejoramiento de imágenes (CLAHE)")
    parser.add_argument("--amp", action="store_true",
                        help="Activar Automatic Mixed Precision")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reanudar desde checkpoint")
    
    # Parámetros de testing
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Umbral de confianza para detecciones")
    parser.add_argument("--no-save-test-images", action="store_true",
                        help="No guardar imágenes con detecciones")
    
    # Control de flujo
    parser.add_argument("--skip-stages", nargs="+", 
                        choices=["split", "augment", "train", "test"],
                        default=[],
                        help="Etapas a saltar")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continuar pipeline aunque una etapa falle")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Crear configuración
    config = {
        "dataset_images": args.dataset_images,
        "dataset_xmls": args.dataset_xmls,
        "test_images": args.test_images,
        "test_xmls": args.test_xmls,
        "augmented_images": args.augmented_images,
        "augmented_xmls": args.augmented_xmls,
        "output_dir": args.output_dir,
        "test_output": args.test_output,
        "test_split": args.test_split,
        "seed": args.seed,
        "num_augmentations": args.num_augmentations,
        "copy_originals": args.copy_originals,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "patience": args.patience,
        "enhance": args.enhance,
        "amp": args.amp,
        "resume": args.resume,
        "confidence": args.confidence,
        "no_save_test_images": args.no_save_test_images,
        "skip_stages": args.skip_stages,
        "continue_on_error": args.continue_on_error
    }
    
    # Ejecutar pipeline
    pipeline = TrainingPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
