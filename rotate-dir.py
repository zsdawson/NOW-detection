from pathlib import Path
import shutil

# CHANGE THIS
DATASET_DIR = Path("output-new-train")

# output folder
OUT_DIR = Path("output-new-train-fixed-270cw-labels")

SPLITS = ["train", "val", "test"]

def fix_yolo_label_90ccw_270cw(label_path, out_label_path, img_w, img_h):
    fixed_lines = []

    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue

            cls, x, y, bw, bh = p
            x, y, bw, bh = map(float, [x, y, bw, bh])

            # labels were made in swapped 90-degree space
            label_w = img_h
            label_h = img_w

            xmin = (x - bw / 2) * label_w
            ymin = (y - bh / 2) * label_h
            xmax = (x + bw / 2) * label_w
            ymax = (y + bh / 2) * label_h

            points = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ]

            # correction for "Labels corrected from 90 CCW / 270 CW image"
            fixed_points = [(img_w - y, x) for x, y in points]

            xs = [pt[0] for pt in fixed_points]
            ys = [pt[1] for pt in fixed_points]

            fxmin = max(0, min(img_w, min(xs)))
            fymin = max(0, min(img_h, min(ys)))
            fxmax = max(0, min(img_w, max(xs)))
            fymax = max(0, min(img_h, max(ys)))

            new_x = ((fxmin + fxmax) / 2) / img_w
            new_y = ((fymin + fymax) / 2) / img_h
            new_bw = (fxmax - fxmin) / img_w
            new_bh = (fymax - fymin) / img_h

            fixed_lines.append(
                f"{cls} {new_x:.6f} {new_y:.6f} {new_bw:.6f} {new_bh:.6f}\n"
            )

    out_label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_label_path, "w") as f:
        f.writelines(fixed_lines)


for split in SPLITS:
    img_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    out_img_dir = OUT_DIR / "images" / split
    out_label_dir = OUT_DIR / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print("Skipping missing split:", split)
        continue

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(img_dir.glob(ext)))

    fixed_count = 0
    missing_label_count = 0

    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"

        shutil.copy(img_path, out_img_dir / img_path.name)

        if not label_path.exists():
            missing_label_count += 1
            continue

        from PIL import Image
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        out_label_path = out_label_dir / label_path.name

        fix_yolo_label_90ccw_270cw(
            label_path,
            out_label_path,
            img_w,
            img_h
        )

        fixed_count += 1

    print(split)
    print("  images:", len(image_paths))
    print("  fixed labels:", fixed_count)
    print("  missing labels:", missing_label_count)

print("Done. Corrected dataset saved to:", OUT_DIR)
