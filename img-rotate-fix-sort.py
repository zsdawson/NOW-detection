from pathlib import Path
import shutil
import csv
import cv2
import numpy as np
from ultralytics import YOLO

DATASET_DIR = Path("output-new-train").resolve()
OUT_DIR = Path("full_dataset_sorted").resolve()

MODEL_PATH = Path("runs/detect/seed_orientation_checker/weights/best.pt").resolve()

if not MODEL_PATH.exists():
    alt = Path("runs/detect/seed_orientation_checker2/weights/best.pt").resolve()
    if alt.exists():
        MODEL_PATH = alt

SPLITS = ["train", "val", "test"]

CONF_THRES = 0.20
AUTO_MARGIN = 0.08
GOOD_IOU_THRES = 0.18
REQUIRE_MODEL_DETECTIONS = True

def read_yolo_boxes(label_path, label_w, label_h):
    boxes = []

    if not label_path.exists():
        return boxes

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
            (xmin, ymax)
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

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union

def score_labels_against_predictions(label_boxes, pred_boxes):
    if len(label_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    scores = []

    for _, xmin, ymin, xmax, ymax in label_boxes:
        label_box = (xmin, ymin, xmax, ymax)
        best = 0.0

        for pred_box in pred_boxes:
            best = max(best, iou(label_box, pred_box))

        scores.append(best)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))

def get_model_predictions(model, img):
    try:
        result = model.predict(
            source=img,
            conf=CONF_THRES,
            verbose=False
        )[0]
    except Exception as e:
        print("Prediction failed:", e)
        return None

    pred_boxes = []

    if result.boxes is None:
        return pred_boxes

    for b in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        pred_boxes.append((float(x1), float(y1), float(x2), float(y2)))

    return pred_boxes

def save_clean(img_path, chosen_boxes, split, img_w, img_h):
    out_img_dir = OUT_DIR / "clean" / "images" / split
    out_lab_dir = OUT_DIR / "clean" / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(img_path, out_img_dir / img_path.name)

    out_label_path = out_lab_dir / f"{img_path.stem}.txt"
    lines = boxes_to_yolo_lines(chosen_boxes, img_w, img_h)

    with open(out_label_path, "w") as f:
        f.writelines(lines)

def save_manual(img_path, label_path, split):
    out_img_dir = OUT_DIR / "manual_review" / "images" / split
    out_lab_dir = OUT_DIR / "manual_review" / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(img_path, out_img_dir / img_path.name)

    if label_path.exists():
        shutil.copy(label_path, out_lab_dir / label_path.name)

print("Dataset:", DATASET_DIR)
print("Model:", MODEL_PATH)
print("Output:", OUT_DIR)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

(OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))
report_rows = []

for split in SPLITS:
    img_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    if not img_dir.exists():
        print("Skipping missing split:", split)
        continue

    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(list(img_dir.glob(ext)))

    counts = {
        "clean_original": 0,
        "clean_270": 0,
        "manual_review": 0,
        "missing_label": 0,
        "bad_image": 0,
        "no_model_predictions": 0,
        "prediction_failed": 0
    }

    print(f"\nProcessing {split}: {len(image_paths)} images")

    for idx, img_path in enumerate(image_paths, 1):
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            counts["missing_label"] += 1
            report_rows.append([
                split,
                img_path.name,
                "missing_label",
                "",
                "",
                "",
                0
            ])
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            counts["bad_image"] += 1
            save_manual(img_path, label_path, split)
            report_rows.append([
                split,
                img_path.name,
                "manual_review_bad_image",
                "",
                "",
                "",
                0
            ])
            continue

        img_h, img_w = img.shape[:2]

        pred_boxes = get_model_predictions(model, img)

        if pred_boxes is None:
            counts["manual_review"] += 1
            counts["prediction_failed"] += 1
            save_manual(img_path, label_path, split)
            report_rows.append([
                split,
                img_path.name,
                "manual_review_prediction_failed",
                0,
                0,
                0,
                0
            ])
            continue

        if REQUIRE_MODEL_DETECTIONS and len(pred_boxes) == 0:
            counts["manual_review"] += 1
            counts["no_model_predictions"] += 1
            save_manual(img_path, label_path, split)
            report_rows.append([
                split,
                img_path.name,
                "manual_review_no_model_predictions",
                0,
                0,
                0,
                0
            ])
            continue

        original_boxes = read_yolo_boxes(label_path, img_w, img_h)
        fixed_270_boxes = fix_270cw_label_boxes(label_path, img_w, img_h)

        original_score = score_labels_against_predictions(original_boxes, pred_boxes)
        fixed_270_score = score_labels_against_predictions(fixed_270_boxes, pred_boxes)

        score_gap = abs(fixed_270_score - original_score)

        if original_score >= GOOD_IOU_THRES and original_score >= fixed_270_score + AUTO_MARGIN:
            save_clean(img_path, original_boxes, split, img_w, img_h)
            counts["clean_original"] += 1
            decision = "clean_original"

        elif fixed_270_score >= GOOD_IOU_THRES and fixed_270_score >= original_score + AUTO_MARGIN:
            save_clean(img_path, fixed_270_boxes, split, img_w, img_h)
            counts["clean_270"] += 1
            decision = "clean_270"

        else:
            save_manual(img_path, label_path, split)
            counts["manual_review"] += 1
            decision = "manual_review_uncertain"

        report_rows.append([
            split,
            img_path.name,
            decision,
            round(original_score, 4),
            round(fixed_270_score, 4),
            round(score_gap, 4),
            len(pred_boxes)
        ])

        if idx % 50 == 0:
            print(f"processed {idx}/{len(image_paths)}")

    print(split, counts)

report_path = OUT_DIR / "reports" / "orientation_report.csv"

with open(report_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "split",
        "filename",
        "decision",
        "original_score",
        "fixed_270_score",
        "score_gap",
        "num_model_predictions"
    ])
    writer.writerows(report_rows)

clean_yaml = OUT_DIR / "clean" / "data.yaml"
clean_yaml.parent.mkdir(parents=True, exist_ok=True)

clean_yaml.write_text(f"""path: {OUT_DIR / "clean"}
train: images/train
val: images/val
test: images/test

names:
  0: NOW
""")

print("\nDone.")
print("Clean dataset:", OUT_DIR / "clean")
print("Manual review:", OUT_DIR / "manual_review")
print("Report:", report_path)
print("Clean YAML:", clean_yaml)

try:
    import pandas as pd
    report = pd.read_csv(report_path)
    print("\nDecision counts:")
    print(report["decision"].value_counts())
except Exception as e:
    print("Could not show pandas report summary:", e)
    
