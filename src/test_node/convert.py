
from ultralytics import YOLO

model = YOLO("/home/nuninu98/Downloads/yolov8n.pt")
model.export(format="onnx", imgsz=[640, 640], opset=12, device=0)