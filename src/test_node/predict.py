from ultralytics import YOLO

model = YOLO("/home/nuninu98/junhyung/runs/segment/train/weights/best.pt")
results = model("/home/nuninu98/junhyung/front/1724894374_753520012.jpg", show=True)