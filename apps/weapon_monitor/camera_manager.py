import cv2
import threading
import time
import queue
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Optional

from weapon_detection.inference.detector_pipeline import WeaponDetectionPipeline
import database

class CameraThread(threading.Thread):
    def __init__(self, camera_config: Dict, pipeline: WeaponDetectionPipeline, alarm_dir: Path):
        super().__init__()
        self.camera_id = camera_config['id']
        self.name = camera_config['name']
        self.source = camera_config['source']
        
        # Convertir source a int si es un número (webcam index)
        try:
            self.source_idx = int(self.source)
        except ValueError:
            self.source_idx = self.source
            
        self.pipeline = pipeline
        self.alarm_dir = alarm_dir
        
        # Configuración
        self.conf_threshold = camera_config['confidence_threshold']
        self.cooldown = camera_config['cooldown']
        self.consecutive_frames_req = camera_config['consecutive_frames']
        
        # Parámetros de rendimiento (configurables por entorno)
        self.stream_width = int(os.getenv('STREAM_WIDTH', '640'))
        self.stream_height = int(os.getenv('STREAM_HEIGHT', '480'))
        self.stream_fps = int(os.getenv('STREAM_FPS', '15'))
        self.skip_frames = int(os.getenv('FRAME_SKIP', '4'))  # Procesar 1 de cada N+1
        self.jpeg_quality = int(os.getenv('JPEG_QUALITY', '75'))
        
        # Estado
        self.running = False
        self.cap = None
        self.last_frame = None
        self.last_detections = []
        self.lock = threading.Lock()
        
        # Lógica de alarma
        self.consecutive_count = 0
        self.last_alarm_time = 0
        self.current_weapon_type = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source_idx)
        
        if not self.cap.isOpened():
            print(f"❌ Error al abrir cámara {self.name} ({self.source})")
            self.running = False
            return

        print(f"✅ Cámara iniciada: {self.name}")
        
        # Optimización: reducir resolución para streaming más fluido
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.stream_fps)
        
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"⚠️  Pérdida de señal en cámara {self.name}")
                break
            
            frame_count += 1
            
            # Inferencia solo cada N frames
            if frame_count % (self.skip_frames + 1) == 0:
                # Copia para dibujar
                annotated_frame = frame.copy()
                
                # 1. Extraer personas y detectar armas
                # Usamos el pipeline pero accedemos a métodos internos para optimizar si fuera necesario
                # O usamos process_image directamente
                _, detections = self.pipeline.process_image(frame)
                
                # Filtrar por confianza
                valid_detections = [
                    d for d in detections 
                    if d['confidence'] >= self.conf_threshold
                ]
                
                # Lógica de consistencia temporal
                if valid_detections:
                    # Asumimos que si hay múltiples, tomamos la de mayor confianza
                    best_det = max(valid_detections, key=lambda x: x['confidence'])
                    
                    if self.current_weapon_type == best_det['weapon_class']:
                        self.consecutive_count += 1
                    else:
                        self.consecutive_count = 1
                        self.current_weapon_type = best_det['weapon_class']
                    
                    # Verificar si disparar alarma
                    if self.consecutive_count >= self.consecutive_frames_req:
                        self._trigger_alarm(frame, best_det)
                else:
                    self.consecutive_count = 0
                    self.current_weapon_type = None
                
                # Re-procesamos visualización para mostrar estado de alarma
                for det in valid_detections:
                    # Dibujar bounding box
                    wx1, wy1, wx2, wy2 = det['weapon_bbox']
                    cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{det['weapon_class']} {det['confidence']:.2f}", 
                               (wx1, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if self.consecutive_count > 0:
                     cv2.putText(annotated_frame, f"ALERTA: {self.consecutive_count}/{self.consecutive_frames_req}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                with self.lock:
                    self.last_frame = annotated_frame
                    self.last_detections = valid_detections
            
            else:
                # Si saltamos inferencia, enviamos frame sin procesar para mantener fluidez
                with self.lock:
                    # Reutilizar anotaciones previas sobre frame actual
                    display_frame = frame.copy()
                    for det in self.last_detections:
                        wx1, wy1, wx2, wy2 = det['weapon_bbox']
                        cv2.rectangle(display_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                        cv2.putText(display_frame, f"{det['weapon_class']} {det['confidence']:.2f}", 
                                   (wx1, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    if self.consecutive_count > 0:
                        cv2.putText(display_frame, f"ALERTA: {self.consecutive_count}/{self.consecutive_frames_req}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    self.last_frame = display_frame
            
            # Sleep mínimo para no saturar CPU
            time.sleep(0.005)
            
        self.cap.release()
        print(f"🛑 Cámara detenida: {self.name}")

    def _trigger_alarm(self, frame, detection):
        now = time.time()
        if now - self.last_alarm_time < self.cooldown:
            return

        print(f"🚨 ALARMA DETECTADA en {self.name}: {detection['weapon_class']}")
        self.last_alarm_time = now
        
        # Generar nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"alarm_{self.camera_id}_{timestamp}"
        
        full_img_name = f"{filename_base}_full.jpg"
        crop_img_name = f"{filename_base}_crop.jpg"
        
        full_path = self.alarm_dir / full_img_name
        crop_path = self.alarm_dir / crop_img_name
        
        # Guardar imagen completa
        # Dibujar caja en la imagen guardada
        save_frame = frame.copy()
        wx1, wy1, wx2, wy2 = detection['weapon_bbox']
        cv2.rectangle(save_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
        cv2.imwrite(str(full_path), save_frame)
        
        # Guardar recorte del arma (o de la persona con el arma)
        # Usemos el recorte del arma con un poco de margen
        h, w = frame.shape[:2]
        margin = 20
        cx1 = max(0, wx1 - margin)
        cy1 = max(0, wy1 - margin)
        cx2 = min(w, wx2 + margin)
        cy2 = min(h, wy2 + margin)
        
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        
        # Guardar en DB
        database.add_alarm(
            camera_id=self.camera_id,
            weapon_type=detection['weapon_class'],
            confidence=detection['confidence'],
            image_path=full_img_name,
            crop_path=crop_img_name
        )

    def get_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None
            # Codificar a JPEG con compresión para reducir latencia
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, buffer = cv2.imencode('.jpg', self.last_frame, encode_param)
            return buffer.tobytes() if ret else None
    
    def stop(self):
        self.running = False
        self.join()

class CameraManager:
    def __init__(self, pipeline: WeaponDetectionPipeline, alarm_dir: Path):
        self.pipeline = pipeline
        self.alarm_dir = alarm_dir
        self.threads: Dict[int, CameraThread] = {}
        
        # Asegurar directorio de alarmas
        self.alarm_dir.mkdir(parents=True, exist_ok=True)
        
    def start_camera(self, camera_id: int):
        if camera_id in self.threads:
            return # Ya está corriendo
            
        config = database.get_camera(camera_id)
        if not config:
            print(f"❌ Cámara {camera_id} no encontrada en DB")
            return
            
        thread = CameraThread(config, self.pipeline, self.alarm_dir)
        thread.start()
        self.threads[camera_id] = thread
        
    def stop_camera(self, camera_id: int):
        if camera_id in self.threads:
            self.threads[camera_id].stop()
            del self.threads[camera_id]
            
    def get_frame(self, camera_id: int):
        if camera_id in self.threads:
            return self.threads[camera_id].get_frame()
        return None
        
    def stop_all(self):
        for cam_id in list(self.threads.keys()):
            self.stop_camera(cam_id)
            
    def reload_cameras(self):
        """Reinicia todas las cámaras habilitadas según DB."""
        self.stop_all()
        cameras = database.get_cameras(only_enabled=True)
        for cam in cameras:
            self.start_camera(cam['id'])
