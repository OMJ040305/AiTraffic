import cv2
import torch
import numpy as np
import matplotlib.path as mplPath


class VehicleDetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            # Usamos 'yolov5n' (nano) para velocidad
            self.model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
            if torch.cuda.is_available():
                self.model.cuda()
                print("🚀 DETECTOR: GPU Activada (CUDA)")
            else:
                print("⚠️ DETECTOR: Usando CPU")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model = None

    def get_center(self, bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def is_valid_detection(self, xc, yc, zones):
        for zone in zones:
            if len(zone) > 0 and mplPath.Path(zone).contains_point((xc, yc)):
                return True
        return False

    def detect(self, frame):
        if self.model is None:
            return []

        height, width = frame.shape[:2]
        scale = 0.5
        small_frame = cv2.resize(frame, (int(width * scale), int(height * scale))) if 'cv2' in globals() else frame

        # Nota: Para mantener este archivo puro sin cv2, asumimos que el frame llega listo
        # o importamos cv2 si queremos redimensionar aquí.
        # Por simplicidad, haremos inferencia directa o asumimos que 'main' gestiona el resize si es crítico.

        preds = self.model(frame)
        df = preds.pandas().xyxy[0]
        df = df[df["confidence"] >= 0.2]
        df = df[df["name"].isin(["car", "truck", "bus", "motorcycle"])]
        return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)