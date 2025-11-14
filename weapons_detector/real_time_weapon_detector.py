#!/usr/bin/env python3
"""
Detección en tiempo real de armas (knife, pistol) usando Faster R-CNN entrenado.

Requisitos:
  - Modelo entrenado: best_model.pth (ver train_fasterrcnn.py)
  - classes.json con mapa de clases
  - pip install torch torchvision opencv-python

Uso:
  python weapons_detector/real_time_weapon_detector.py --model-path results_frcnn/best_model.pth --classes-path results_frcnn/classes.json
  python weapons_detector/real_time_weapon_detector.py --video input.mp4

Teclas:
  q -> salir

Si se desea integrar con Stage1 (detección de personas), se puede usar la salida de recortes de personas como fuente de frames.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import torch
import torchvision
from typing import cast
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEFAULT_THRESHOLD = 0.6


def load_model(model_path: Path, classes_path: Path, device: torch.device):
    data = json.loads(classes_path.read_text())
    class_map = data['classes']  # {"knife":1,"pistol":2}
    num_classes = len(class_map) + 1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    # Extraer in_features del predictor original
    original_predictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    in_features = original_predictor.cls_score.in_features  # type: ignore[attr-defined]
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    inv_class_map = {v: k for k, v in class_map.items()}
    return model, inv_class_map


def draw_detections(frame, boxes, labels, scores, inv_class_map, threshold):
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_name = inv_class_map.get(int(label), str(label))
        color = (0, 0, 255) if cls_name == 'knife' else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name}:{score:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def process_frame(model, frame, device):
    # Convertir a tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2,0,1).to(torch.float32)/255.0
    with torch.no_grad():
        out = model([tensor.to(device)])[0]
    return out


def run_video_loop(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, inv_class_map = load_model(Path(args.model_path), Path(args.classes_path), device)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir fuente de video")

    print("Presiona 'q' para salir.")
    last_time = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        out = process_frame(model, frame, device)
        frame = draw_detections(frame, out['boxes'], out['labels'], out['scores'], inv_class_map, args.threshold)
        # FPS simple
        if frames % 10 == 0:
            now = time.time()
            fps = 10 / (now - last_time)
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),1)
        cv2.imshow('Weapon Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_args():
    ap = argparse.ArgumentParser(description='Detección tiempo real Faster R-CNN armas')
    ap.add_argument('--model-path', required=True, help='Ruta a best_model.pth')
    ap.add_argument('--classes-path', required=True, help='Ruta a classes.json')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--video', type=str, default='', help='Ruta a video (vacío = webcam)')
    ap.add_argument('--device', type=str, default='cuda')
    return ap.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_video_loop(args)
