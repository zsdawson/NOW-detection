from ultralytics import YOLO

DATA_YAML = "/path/to/yolo_dataset/dataset.yaml"

model = YOLO("yolov8m.pt")

results = model.train(
    data=DATA_YAML,
    epochs=200,
    imgsz=960,
    batch=-1,          # auto batch based on GPU memory
    device=0,          # A100
    patience=40,
    optimizer="auto",
    cos_lr=True,
    close_mosaic=15,
    amp=True,
    cache="disk",
    workers=8,
    pretrained=True,
    project="runs_yolo",
    name="insect_yolov8m_960",
    exist_ok=True,
    seed=42,
    plots=True,
)
