# machine learning for behavioural detection of Pangolin Sunda
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="data_custom.yaml", imgsz = 640, batch = 8, epochs = 100, workers = 0, device = 0)
