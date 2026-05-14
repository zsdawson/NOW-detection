from pathlib import Path
import shutil
import random
import cv2
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================

DATASET_DIR = Path("output-new-train").resolve()
IMG_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

OUT_DIR = Path("seed_good").resolve()
TRAIN_RATIO = 0.8

MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 960
BATCH = 4

# =========================
# YOUR REVIEW RESULTS
# =========================

zero_rotation = {
    "S2T6 07-03-2024.jpeg",
    "S1T3 10-02-2024.jpeg",
    "L2T1 05-29-2024.jpg",
    "image1.jpg",
}

bad_skip = {
    "S2T5 10-09-2024.jpg",
    "S1T6 10-09-2024.jpg",
}

sample_files = [
    "S1T1 06-12-2024.jpg",
    "N3T1 03-12-2025.jpg",
    "S1T1 11-06-2024.jpg",
    "S1T8 06-12-2024.jpg",
    "L1T7 08-07-2024.jpg",
    "L2T3  01-22-2025.jpg",
    "L2T5 11-06-2024.jpg",
    "L2T1 08-28-2024.jpg",
    "S1T4 03-12-2025.jpg",
    "S2T6 07-03-2024.jpeg",
    "L1T2 10-09-2024.jpg",
    "S1T6 06-12-2024.jpg",
    "S2T5 10-09-2024.jpg",
    "S1T3 10-02-2024.jpeg",
    "CT1 03-12-2025.jpg",
    "S2T6 06-12-2024.jpg",
    "S2T8 08-07-2024.jpg",
    "L2T8  01-22-2025.jpg",
    "N3T7 03-12-2025.jpg",
    "CT2 03-12-2025.jpg",
    "L2T4 10-30-2024.jpg",
    "S1T4 10-23-2024.jpg",
    "S1T3 10-30-2024.jpg",
    "L1T8 7-10-24.jpg",
    "S1T6 10-23-2024.jpg",
    "S1T4 08-07-2024.jpg",
    "L1T6 06-19-2024.jpg",
    "S2T4 03-04-25.jpg",
    "L2T3 10-09-2024.jpg",
    "S2T1 08-21-2024.jpg",
    "S1T2 05-22-2024.jpg",
    "L2T1 05-29-2024.jpg",
    "L2T6 08-07-2024.jpg",
    "S2T4 08-14-2024.jpg",
    "image1.jpg",
    "L1T8 06-12-2024.jpg",
    "S1T5 03-12-2025.jpg",
    "S1T7 05-22-2024.jpg",
    "L1T1 06-19-2024.jpg",
    "L1T7 06-12-2024.jpg",
    "L1T7 03-04-25.jpg",
    "S2T8 09-11-2024.jpg",
    "S2T7 06-19-2024.jpg",
    "S2T6 10-16-2024.jpg",
    "S2T3 08-07-2024.jpg",
    "S2T3 7-10-24.jpg",
    "S1T4 06-26+27-2024.jpg",
    "S1T6 10-09-2024.jpg",
    "S2T8 08-14-2024.jpg",
    "L1T6 08-07-2024.jpg",
]

# =========================
# FUNCTIONS
# =========================

def read_yolo_boxes(label_path, label_w, label_h):
    boxes = []

    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()

            if len(p) != 5:
                continue

            cls, x, y, bw, bh = p
            x, y, bw, bh = map(float, [x, y, bw, bh])

            xmin = (x - bw / 2) * label_w
            ymin = (y - bh / 2) * label_h
            xmax = (x + bw / 2) * label_w
            ymax = (y + bh / 2) * label_h

            boxes.append((cls, xmin, ymin, xmax, ymax))

    return boxes


def fix_270cw_label_boxes(label_path, img_w, img_h):
    raw_boxes = read_yolo_boxes(label_path, img_h, img_w)
    fixed_boxes = []

    for cls, xmin, ymin, xmax, ymax in raw_boxes:
        points = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]

        fixed_points = [(img_w - y, x) for x, y in points]

        xs = [p[0] for p in fixed_points]
        ys = [p[1] for p in fixed_points]

        fxmin = max(0, min(img_w, min(xs)))
        fymin = max(0, min(img_h, min(ys)))
        fxmax = max(0, min(img_w, max(xs)))
        fymax = max(0, min(img_h, max(ys)))

        fixed_boxes.append((cls, fxmin, fymin, fxmax, fymax))

    return fixed_boxes


def boxes_to_yolo_lines(boxes, img_w, img_h):
    lines = []

    for cls, xmin, ymin, xmax, ymax in boxes:
        xmin = max(0, min(img_w, xmin))
        ymin = max(0, min(img_h, ymin))
        xmax = max(0, min(img_w, xmax))
        ymax = max(0, min(img_h, ymax))

        bw = xmax - xmin
        bh = ymax - ymin

        if bw <= 1 or bh <= 1:
            continue

        x = ((xmin + xmax) / 2) / img_w
        y = ((ymin + ymax) / 2) / img_h
        bw = bw / img_w
        bh = bh / img_h

        lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    return lines


# =========================
# BUILD SEED DATASET
# =========================

print("Dataset folder:", DATASET_DIR)
print("Image folder:", IMG_DIR)
print("Label folder:", LABEL_DIR)

if not IMG_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label folder not found: {LABEL_DIR}")

if OUT_DIR.exists():
    print("Deleting old seed_good folder...")
    shutil.rmtree(OUT_DIR)

for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

usable_files = [f for f in sample_files if f not in bad_skip]
random.shuffle(usable_files)

split_idx = int(len(usable_files) * TRAIN_RATIO)
train_files = set(usable_files[:split_idx])

copied = 0
kept_original = 0
fixed_270 = 0
skipped_bad = 0
missing = []

for filename in sample_files:
    if filename in bad_skip:
        skipped_bad += 1
        continue

    img_path = IMG_DIR / filename
    label_path = LABEL_DIR / f"{Path(filename).stem}.txt"

    if not img_path.exists() or not label_path.exists():
        missing.append(filename)
        continue

    split = "train" if filename in train_files else "val"

    out_img = OUT_DIR / "images" / split / filename
    out_label = OUT_DIR / "labels" / split / f"{Path(filename).stem}.txt"

    shutil.copy(img_path, out_img)

    img = cv2.imread(str(img_path))

    if img is None:
        missing.append(filename)
        continue

    img_h, img_w = img.shape[:2]

    if filename in zero_rotation:
        shutil.copy(label_path, out_label)
        kept_original += 1
    else:
        fixed_boxes = fix_270cw_label_boxes(label_path, img_w, img_h)
        fixed_lines = boxes_to_yolo_lines(fixed_boxes, img_w, img_h)

        with open(out_label, "w") as f:
            f.writelines(fixed_lines)

        fixed_270 += 1

    copied += 1

# =========================
# WRITE DATA.YAML WITH ABSOLUTE PATH
# =========================

data_yaml = OUT_DIR / "data.yaml"

data_yaml.write_text(f"""path: {OUT_DIR}
train: images/train
val: images/val

names:
  0: NOW
""")

print("\nSeed dataset created.")
print("Copied usable images:", copied)
print("Kept original labels:", kept_original)
print("Fixed 270 labels:", fixed_270)
print("Skipped bad:", skipped_bad)
print("Missing:", len(missing))

if missing:
    print("\nMissing files:")
    for f in missing:
        print(" -", f)

print("\nData YAML:")
print(data_yaml.read_text())

print("Train images:", len(list((OUT_DIR / "images" / "train").glob("*"))))
print("Val images:", len(list((OUT_DIR / "images" / "val").glob("*"))))

if len(list((OUT_DIR / "images" / "val").glob("*"))) == 0:
    raise RuntimeError("No validation images were created. Training will fail.")

# =========================
# TRAIN MINI MODEL
# =========================

print("\nStarting mini-model training...")

model = YOLO(MODEL_NAME)

model.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    name="seed_orientation_checker",
)

print("\nDone.")
print("Mini model should be in:")
print("runs/detect/seed_orientation_checker/weights/best.pt")
