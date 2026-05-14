from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="seed_good/data.yaml",
    epochs=50,
    imgsz=960,
    batch=4,
    name="seed_orientation_checker"
)
