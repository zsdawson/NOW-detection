import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

IMG_DIR = Path("output-new-train/images/test")
LABEL_DIR = Path("output-new-train/labels/test")
NUM_SAMPLES = 4

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

def box_from_points(cls, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return cls, int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def fix_labels_from_90cw_image(box, w, h):
    # label was made on image rotated 90 CW
    cls, xmin, ymin, xmax, ymax = box

    points = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax)
    ]

    # inverse of 90 CW image rotation
    fixed = [(y, h - x) for x, y in points]

    return box_from_points(cls, fixed)

def fix_labels_from_90ccw_image(box, w, h):
    # label was made on image rotated 90 CCW / 270 CW
    cls, xmin, ymin, xmax, ymax = box

    points = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax)
    ]

    # inverse of 90 CCW image rotation
    fixed = [(w - y, x) for x, y in points]

    return box_from_points(cls, fixed)

def draw_boxes(img, boxes):
    img = img.copy()
    h, w = img.shape[:2]

    for cls, xmin, ymin, xmax, ymax in boxes:
        xmin = max(0, min(w - 1, int(xmin)))
        ymin = max(0, min(h - 1, int(ymin)))
        xmax = max(0, min(w - 1, int(xmax)))
        ymax = max(0, min(h - 1, int(ymax)))

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (xmin, max(20, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(list(IMG_DIR.glob(ext)))

valid_pairs = []
for img_path in image_paths:
    label_path = LABEL_DIR / f"{img_path.stem}.txt"
    if label_path.exists():
        valid_pairs.append((img_path, label_path))

print("Pairs found:", len(valid_pairs))

samples = random.sample(valid_pairs, min(NUM_SAMPLES, len(valid_pairs)))

for img_path, label_path in samples:
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # normal YOLO label read
    original_boxes = read_yolo_boxes(label_path, w, h)

    # important: swapped dimensions for 90-degree label space
    rotated_label_boxes = read_yolo_boxes(label_path, h, w)

    fixed_from_90cw = [fix_labels_from_90cw_image(b, w, h) for b in rotated_label_boxes]
    fixed_from_90ccw = [fix_labels_from_90ccw_image(b, w, h) for b in rotated_label_boxes]

    imgs = [
        draw_boxes(img, original_boxes),
        draw_boxes(img, fixed_from_90cw),
        draw_boxes(img, fixed_from_90ccw),
    ]

    titles = [
        "Original labels",
        "Labels corrected from 90 CW image",
        "Labels corrected from 90 CCW / 270 CW image",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(img_path.name)
    plt.tight_layout()
    plt.show()
