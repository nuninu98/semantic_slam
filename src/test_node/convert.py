
from ultralytics import YOLO
#from roboflow import Roboflow

                
#model = YOLO("yolov8n-seg.pt")
#model.export(format="onnx", imgsz=[640, 640], opset=12, device=0)
onnx_model = YOLO("/home/nuninu98/catkin_ws/src/semantic_slam/model/yolov8n.onnx")
result = onnx_model("/home/nuninu98/Pictures/Screenshot from 2024-09-05 15-07-43.png")
for res in result:
    res.show()
#model.export(format="onnx", imgsz=[640, 640], opset=12, device=0)