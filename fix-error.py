from pathlib import Path
from ultralytics import YOLO

data_yaml = Path("seed_good/data.yaml").resolve()

print("Using data.yaml:", data_yaml)
print(data_yaml.read_text())

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=960,
    batch=4,
    workers=1,              # avoids your dataloader warning
    name="seed_orientation_checker",
    exist_ok=True           # lets it reuse/overwrite same run name
)
