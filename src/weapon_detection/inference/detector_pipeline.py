#!/usr/bin/env python3
"""
Pipeline Completo de Detección de Armas
Integra: Extracción de Personas → Mejora de Imagen → Detección de Armas
"""

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional, Callable
import subprocess
import os

# Configuración
PERSON_CLASS_ID = 0
WEAPON_CLASSES = ["__background__", "knife", "pistol"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "weapon_detection" / "best_model.pth"
DEFAULT_YOLO_PATH = REPO_ROOT / "src" / "person_extraction" / "yolov8n.pt"


class WeaponDetectionPipeline:
    """Pipeline completo para detectar armas en videos o imágenes."""
    
    def __init__(
        self,
        weapon_model_path: str = str(DEFAULT_MODEL_PATH),
        yolo_model_path: str = str(DEFAULT_YOLO_PATH),
        confidence_threshold: float = 0.5,
        min_person_conf: float = 0.5
    ):
        """
        Inicializa el pipeline.
        
        Args:
            weapon_model_path: Ruta al modelo de detección de armas
            yolo_model_path: Ruta al modelo YOLO para personas
            confidence_threshold: Umbral de confianza para detección de armas
            min_person_conf: Umbral de confianza mínimo para detección de personas
        """
        self.confidence_threshold = confidence_threshold
        self.min_person_conf = min_person_conf
        
        # Cargar modelo YOLO para personas
        print("🔄 Cargando modelo YOLO para detección de personas...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Cargar modelo de detección de armas
        print("🔄 Cargando modelo de detección de armas...")
        self.weapon_model = self._load_weapon_model(weapon_model_path)
        self.weapon_model.eval()
        
        print(f"✅ Pipeline inicializado (Dispositivo: {DEVICE})")
    
    def _load_weapon_model(self, model_path: str):
        """Carga el modelo Faster R-CNN de detección de armas."""
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, len(WEAPON_CLASSES)
            )
        )

        checkpoint = torch.load(model_path, map_location=DEVICE)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Quitar el prefijo `module.` si el modelo se entrenó con DDP/DataParallel
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Pesos faltantes en el checkpoint: {len(missing)} entradas")
        if unexpected:
            print(f"⚠️  Pesos inesperados ignorados: {len(unexpected)} entradas")

        model.to(DEVICE)
        return model
    
    def extract_persons(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extrae recortes de todas las personas detectadas en un frame.
        
        Args:
            frame: Imagen BGR (numpy array)
            
        Returns:
            Lista de tuplas (recorte_persona, (x1, y1, x2, y2))
        """
        results = self.yolo_model(frame, verbose=False)
        persons = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == PERSON_CLASS_ID and confidence > self.min_person_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Asegurar límites
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size > 0:
                            persons.append((person_crop, (x1, y1, x2, y2)))
        
        return persons
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Mejora la calidad de la imagen (contraste y brillo).
        
        Args:
            image: Imagen BGR
            
        Returns:
            Imagen mejorada
        """
        # Redimensionar si es muy pequeña
        h, w = image.shape[:2]
        min_height, min_width = 200, 100
        
        if h < min_height or w < min_width:
            scale_h = min_height / h if h < min_height else 1.0
            scale_w = min_width / w if w < min_width else 1.0
            scale = max(scale_h, scale_w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Mejorar contraste (CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Mejorar brillo
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        v = np.clip(v, 0, 255)
        enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def detect_weapons(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detecta armas en una imagen.
        
        Args:
            image: Imagen BGR
            
        Returns:
            Lista de tuplas (clase, confianza, (x1, y1, x2, y2))
        """
        # Convertir BGR a RGB y a tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).to(DEVICE)
        
        with torch.no_grad():
            predictions = self.weapon_model([image_tensor])
        
        detections = []
        for box, label, score in zip(
            predictions[0]["boxes"],
            predictions[0]["labels"],
            predictions[0]["scores"]
        ):
            if score >= self.confidence_threshold:
                class_name = WEAPON_CLASSES[label.item()]
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append((class_name, float(score), (x1, y1, x2, y2)))
        
        return detections
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Procesa una imagen completa: extrae personas, mejora y detecta armas.
        
        Args:
            image: Imagen BGR
            
        Returns:
            Tupla (imagen_anotada, lista_detecciones)
        """
        result_image = image.copy()
        all_detections = []
        
        # Extraer personas
        persons = self.extract_persons(image)
        
        if not persons:
            return result_image, []
        
        # Procesar cada persona
        for person_crop, (px1, py1, px2, py2) in persons:
            # Mejorar imagen de la persona
            enhanced_person = self.enhance_image(person_crop)
            
            # Detectar armas en la persona
            weapons = self.detect_weapons(enhanced_person)
            
            if weapons:
                # Dibujar bounding box de la persona (amarillo)
                cv2.rectangle(result_image, (px1, py1), (px2, py2), (0, 255, 255), 3)
                cv2.putText(
                    result_image,
                    "PERSON WITH WEAPON",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
                # Dibujar detecciones de armas dentro del recorte de persona
                for weapon_class, conf, (wx1, wy1, wx2, wy2) in weapons:
                    # Ajustar coordenadas al frame original
                    abs_x1 = px1 + wx1
                    abs_y1 = py1 + wy1
                    abs_x2 = px1 + wx2
                    abs_y2 = py1 + wy2
                    
                    # Dibujar arma (rojo)
                    cv2.rectangle(result_image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                    text = f"{weapon_class}: {conf:.2f}"
                    cv2.putText(
                        result_image,
                        text,
                        (abs_x1, abs_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                    
                    all_detections.append({
                        "person_bbox": (px1, py1, px2, py2),
                        "weapon_class": weapon_class,
                        "confidence": conf,
                        "weapon_bbox": (abs_x1, abs_y1, abs_x2, abs_y2)
                    })
        
        return result_image, all_detections
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        frame_skip: int = 2,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> Tuple[Optional[str], int, int]:
        """
        Procesa un video completo.
        
        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta opcional para guardar el video procesado
            frame_skip: Procesar 1 de cada N frames (para acelerar)
            progress_callback: Función callback(current_frame, total_frames, detections) para reportar progreso
            
        Returns:
            Tupla (ruta_video_salida, total_frames_procesados, total_detecciones)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurar escritura de video si se especifica
        writer = None
        if output_path:
                # Usar X264 para mejor compatibilidad con navegadores
                # Si falla, usar mp4v como fallback
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'X264')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    # Verificar que el writer se abrió correctamente
                    if not writer.isOpened():
                        print("⚠️  X264 no disponible, usando mp4v")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                except Exception as e:
                    print(f"⚠️  Error con X264: {e}, usando mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        total_detections = 0
        
        print(f"📹 Procesando video: {total_frames} frames @ {fps} FPS")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Procesar cada frame_skip frames
                if frame_count % frame_skip == 0:
                    annotated_frame, detections = self.process_image(frame)
                    processed_count += 1
                    total_detections += len(detections)
                    
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Reportar progreso mediante callback
                    if progress_callback:
                        progress_callback(processed_count, total_frames//frame_skip, total_detections)
                    
                    # Mostrar progreso cada 30 frames
                    if processed_count % 30 == 0:
                        print(f"   Procesados: {processed_count}/{total_frames//frame_skip} frames, "
                              f"Detecciones: {total_detections}")
                else:
                    if writer:
                        writer.write(frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()

            print(f"✅ Video procesado: {processed_count} frames, {total_detections} detecciones")

        # Convertir a H.264 si ffmpeg está disponible (para compatibilidad web)
        if output_path and os.path.exists(output_path):
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            try:
                print("🔄 Convirtiendo a formato web (H.264)...")
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', output_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    temp_output
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0 and os.path.exists(temp_output):
                    # Reemplazar el original con la versión convertida
                    os.replace(temp_output, output_path)
                    print("✅ Video convertido a H.264 para navegadores")
                else:
                    print(f"⚠️  No se pudo convertir (ffmpeg): {result.stderr[:200]}")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"⚠️  ffmpeg no disponible o timeout: {e}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        return output_path, processed_count, total_detections


def main():
    """Función de prueba del pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline de detección de armas")
    parser.add_argument("--input", "-i", required=True, help="Ruta a imagen o video")
    parser.add_argument("--output", "-o", required=True, help="Ruta de salida")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("--frame-skip", type=int, default=1, help="Procesar 1 de cada N frames")
    
    args = parser.parse_args()
    
    # Inicializar pipeline
    pipeline = WeaponDetectionPipeline(confidence_threshold=args.confidence)
    
    input_path = Path(args.input)
    
    # Determinar si es imagen o video
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if input_path.suffix.lower() in image_exts:
        print("📸 Procesando imagen...")
        image = cv2.imread(str(input_path))
        result, detections = pipeline.process_image(image)
        cv2.imwrite(args.output, result)
        print(f"✅ Imagen guardada: {args.output}")
        print(f"   Detecciones: {len(detections)}")
        
    elif input_path.suffix.lower() in video_exts:
        print("📹 Procesando video...")
        pipeline.process_video(str(input_path), args.output, args.frame_skip)
    else:
        print(f"❌ Formato no soportado: {input_path.suffix}")


if __name__ == "__main__":
    main()
