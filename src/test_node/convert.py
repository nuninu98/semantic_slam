
from ultralytics import YOLO


model = YOLO("/home/nuninu98/signage_detection.v1i.yolov8/best.pt")
model.export(format="onnx", imgsz=[640, 640], opset=12)