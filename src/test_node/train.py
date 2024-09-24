from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

model = YOLO("yolov8n.pt")
model.train(data='/home/nuninu98/signage_detection.v1i.yolov8/data.yaml', epochs=50, imgsz=640, batch=8, device=0)


