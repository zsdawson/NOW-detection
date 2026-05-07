import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# USE FOLDERS — NOT SINGLE FILES
IMG_DIR = Path("output-new-train/images/test")
LABEL_DIR = Path("output-new-train/labels/test")

NUM_SAMPLES = 4


def yolo_to_boxes(label_path, w, h):
    boxes = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, x, y, bw, bh = parts
            x = float(x)
            y = float(y)
            bw = float(bw)
            bh = float(bh)

            xmin = int((x - bw / 2) * w)
            ymin = int((y - bh / 2) * h)
            xmax = int((x + bw / 2) * w)
            ymax = int((y + bh / 2) * h)

            boxes.append((cls, xmin, ymin, xmax, ymax))

    return boxes


def rotate_box_90cw(box, w, h):
    cls, xmin, ymin, xmax, ymax = box
    return (
        cls,
        h - ymax,
        xmin,
        h - ymin,
        xmax
    )


def rotate_box_270cw(box, w, h):
    cls, xmin, ymin, xmax, ymax = box
    return (
        cls,
        ymin,
        w - xmax,
        ymax,
        w - xmin
    )


def draw_boxes(img, boxes):
    img = img.copy()

    for cls, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            str(cls),
            (xmin, max(20, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img


image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(list(IMG_DIR.glob(ext)))

valid_pairs = []

for img_path in image_paths:
    label_path = LABEL_DIR / f"{img_path.stem}.txt"

    if label_path.exists():
        valid_pairs.append((img_path, label_path))

print("Images found:", len(image_paths))
print("Image + label pairs found:", len(valid_pairs))

if len(valid_pairs) == 0:
    print("No matching image/label pairs found. Check your folder paths.")

else:
    samples = random.sample(
        valid_pairs,
        min(NUM_SAMPLES, len(valid_pairs))
    )

    for img_path, label_path in samples:
        print("Showing:", img_path.name)

        img = cv2.imread(str(img_path))

        if img is None:
            print("Could not read image:", img_path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        boxes = yolo_to_boxes(label_path, w, h)

        original = draw_boxes(img, boxes)

        boxes_90 = [
            rotate_box_90cw(b, w, h)
            for b in boxes
        ]

        boxes_270 = [
            rotate_box_270cw(b, w, h)
            for b in boxes
        ]

        img_90 = draw_boxes(img, boxes_90)
        img_270 = draw_boxes(img, boxes_270)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        axes[0].imshow(original)
        axes[0].set_title("Original")

        axes[1].imshow(img_90)
        axes[1].set_title("Rotate Boxes 90 CW")

        axes[2].imshow(img_270)
        axes[2].set_title("Rotate Boxes 270 CW")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
