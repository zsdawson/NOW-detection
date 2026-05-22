from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("full_dataset_sorted/clean/data.yaml").resolve()

MODEL_BASE = "yolov8s.pt"   # use "yolov8n.pt" for faster, "yolov8m.pt" for stronger
RUN_NAME = "now_clean_orientation_fixed_yolov8s"

EPOCHS = 150
IMG_SIZE = 960
BATCH = 8                  # lower to 4 if CUDA memory crashes
WORKERS = 1

print("Using dataset:")
print(DATA_YAML)
print(DATA_YAML.read_text())

model = YOLO(MODEL_BASE)

train_results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=WORKERS,
    name=RUN_NAME,
    patience=30,
    cache=False,
    plots=True,
    save=True
)

best_model_path = Path("runs/detect") / RUN_NAME / "weights" / "best.pt"

if not best_model_path.exists():
    matches = list(Path("runs/detect").glob(f"{RUN_NAME}*/weights/best.pt"))
    if matches:
        best_model_path = matches[-1]

print("\nBest model:")
print(best_model_path.resolve())

best_model = YOLO(str(best_model_path))

print("\nRunning validation...")
val_metrics = best_model.val(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    batch=BATCH,
    workers=WORKERS,
    conf=0.25,
    iou=0.7,
    split="val",
    name=RUN_NAME + "_val_eval"
)

print("\nPredicting validation images...")
best_model.predict(
    source=str(Path("full_dataset_sorted/clean/images/val").resolve()),
    imgsz=IMG_SIZE,
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=True,
    name=RUN_NAME + "_val_predictions"
)

test_img_dir = Path("full_dataset_sorted/clean/images/test").resolve()

if test_img_dir.exists() and len(list(test_img_dir.glob("*"))) > 0:
    print("\nRunning test evaluation...")
    test_metrics = best_model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        conf=0.25,
        iou=0.7,
        split="test",
        name=RUN_NAME + "_test_eval"
    )

    print("\nPredicting test images...")
    best_model.predict(
        source=str(test_img_dir),
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        name=RUN_NAME + "_test_predictions"
    )
else:
    print("\nNo test image folder found or test folder is empty. Skipping test.")

print("\nDone.")
print("Training folder:")
print((Path("runs/detect") / RUN_NAME).resolve())

print("\nValidation predictions:")
print((Path("runs/detect") / (RUN_NAME + "_val_predictions")).resolve())

print("\nTest predictions, if created:")
print((Path("runs/detect") / (RUN_NAME + "_test_predictions")).resolve())
