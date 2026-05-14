import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# CHANGE THESE
IMG_DIR = Path("output-new-train/images/train")
LABEL_DIR = Path("output-new-train/labels/train")

NUM_SAMPLES = 50

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


def draw_boxes(img, boxes):
    img = img.copy()
    h, w = img.shape[:2]

    for cls, xmin, ymin, xmax, ymax in boxes:
        xmin = int(max(0, min(w - 1, xmin)))
        ymin = int(max(0, min(h - 1, ymin)))
        xmax = int(max(0, min(w - 1, xmax)))
        ymax = int(max(0, min(h - 1, ymax)))

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


# gather images
image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_paths.extend(list(IMG_DIR.glob(ext)))

pairs = []

for img_path in image_paths:
    label_path = LABEL_DIR / f"{img_path.stem}.txt"

    if label_path.exists():
        pairs.append((img_path, label_path))

print("Images found:", len(image_paths))
print("Image/label pairs found:", len(pairs))

if len(pairs) == 0:
    print("No pairs found. Check paths.")

else:
    samples = random.sample(
        pairs,
        min(NUM_SAMPLES, len(pairs))
    )

    print("\nCOPY THIS LIST:\n")

    file_list = []

    for i, (img_path, label_path) in enumerate(samples, 1):
        print(f"{i}. {img_path.name}")
        file_list.append(img_path.name)

    print("\n--- REVIEW STARTING ---\n")

    for img_path, label_path in samples:
        img = cv2.imread(str(img_path))

        if img is None:
            print("Could not read:", img_path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]

        original_boxes = read_yolo_boxes(
            label_path,
            img_w,
            img_h
        )

        fixed_270_boxes = fix_270cw_label_boxes(
            label_path,
            img_w,
            img_h
        )

        original_img = draw_boxes(img, original_boxes)
        fixed_img = draw_boxes(img, fixed_270_boxes)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 7)
        )

        axes[0].imshow(original_img)
        axes[0].set_title("Original Labels")
        axes[0].axis("off")

        axes[1].imshow(fixed_img)
        axes[1].set_title("270-Corrected Labels")
        axes[1].axis("off")

        plt.suptitle(img_path.name)
        plt.tight_layout()
        plt.show()
