
from ultralytics import YOLO
#from roboflow import Roboflow

                
model = YOLO("yolov8n-seg.pt")
model.export(format="onnx", imgsz=[640, 640], opset=12, device=0)