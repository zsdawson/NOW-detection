from pathlib import Path
import shutil
import random
import csv
import xml.etree.ElementTree as ET
from collections import Counter




# USER SETTINGS
MERGED_DATASET_DIR = Path("/path/to/merged_dataset")

# output folder
YOLO_OUTPUT_DIR = Path("/path/to/yolo_dataset")

# Set True to rebuild YOLO output each run
OVERWRITE_OUTPUT = True

# train ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducible random split
RANDOM_SEED = 42

# One-class detection
CLASS_NAMES = ["NOW"]   # class 0 = insect

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}




# BASIC CHECKS
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

images_src = MERGED_DATASET_DIR / "images"
ann_src = MERGED_DATASET_DIR / "annotations"

if not images_src.exists():
    raise FileNotFoundError(f"Missing images folder: {images_src}")
if not ann_src.exists():
    raise FileNotFoundError(f"Missing annotations folder: {ann_src}")

if YOLO_OUTPUT_DIR.exists():
    if OVERWRITE_OUTPUT:
        shutil.rmtree(YOLO_OUTPUT_DIR)
    else:
        raise FileExistsError(f"{YOLO_OUTPUT_DIR} already exists. Set OVERWRITE_OUTPUT=True")

# YOLO folder structure
for split in ["train", "val", "test"]:
    (YOLO_OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

reports_dir = YOLO_OUTPUT_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# HELPERS

def parse_voc_xml(xml_path):
    """
    Returns:
        width (int), height (int), boxes (list of tuples)
    where boxes are (xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> found in {xml_path}")

    width = int(float(size.findtext("width", default="0")))
    height = int(float(size.findtext("height", default="0")))

    if width <= 0 or height <= 0:
        raise ValueError(f"Bad image size in {xml_path}: width={width}, height={height}")

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))

        # clamp
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append((xmin, ymin, xmax, ymax))

    return width, height, boxes


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h

    # final clamp just in case
    x_center = max(0.0, min(x_center, 1.0))
    y_center = max(0.0, min(y_center, 1.0))
    box_w = max(0.0, min(box_w, 1.0))
    box_h = max(0.0, min(box_h, 1.0))

    return x_center, y_center, box_w, box_h


def find_image_for_stem(images_folder, stem):
    for ext in IMAGE_EXTS:
        p = images_folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_yolo_label(label_path, yolo_lines):
    with label_path.open("w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")





# COLLECT MATCHED FILES
xml_files = sorted(ann_src.glob("*.xml"))
records = []
bad_xml_rows = []
total_objects = 0

for xml_path in xml_files:
    stem = xml_path.stem
    img_path = find_image_for_stem(images_src, stem)

    if img_path is None:
        bad_xml_rows.append({
            "stem": stem,
            "issue": "missing_image",
            "xml_path": str(xml_path),
        })
        continue

    try:
        img_w, img_h, boxes = parse_voc_xml(xml_path)
    except Exception as e:
        bad_xml_rows.append({
            "stem": stem,
            "issue": f"xml_parse_error: {e}",
            "xml_path": str(xml_path),
        })
        continue

    yolo_lines = []
    for (xmin, ymin, xmax, ymax) in boxes:
        x_center, y_center, box_w, box_h = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, img_w, img_h
        )
        # one-class dataset, class id = 0
        yolo_lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    total_objects += len(yolo_lines)

    records.append({
        "stem": stem,
        "image_path": img_path,
        "xml_path": xml_path,
        "object_count": len(yolo_lines),
        "yolo_lines": yolo_lines,
    })

print(f"Usable matched records: {len(records)}")
print(f"Total labeled objects: {total_objects}")
print(f"Bad XML / skipped rows: {len(bad_xml_rows)}")

if len(records) == 0:
    raise ValueError("No usable records found.")



# SPLIT TRAIN / VAL / TEST
random.seed(RANDOM_SEED)
random.shuffle(records)

n = len(records)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_records = records[:n_train]
val_records = records[n_train:n_train + n_val]
test_records = records[n_train + n_val:]

print("\nSplit sizes")
print("-----------")
print(f"Train: {len(train_records)}")
print(f"Val:   {len(val_records)}")
print(f"Test:  {len(test_records)}")


# COPY IMAGES + WRITE LABELS
def export_split(split_name, split_records):
    split_rows = []
    for rec in split_records:
        img_src = rec["image_path"]
        img_dst = YOLO_OUTPUT_DIR / "images" / split_name / img_src.name
        label_dst = YOLO_OUTPUT_DIR / "labels" / split_name / f"{rec['stem']}.txt"

        shutil.copy2(img_src, img_dst)
        write_yolo_label(label_dst, rec["yolo_lines"])

        split_rows.append({
            "split": split_name,
            "stem": rec["stem"],
            "image_name": img_src.name,
            "label_name": f"{rec['stem']}.txt",
            "object_count": rec["object_count"],
            "source_image_path": str(rec["image_path"]),
            "source_xml_path": str(rec["xml_path"]),
        })
    return split_rows

all_export_rows = []
all_export_rows.extend(export_split("train", train_records))
all_export_rows.extend(export_split("val", val_records))
all_export_rows.extend(export_split("test", test_records))


# YAML FILE
yaml_path = YOLO_OUTPUT_DIR / "dataset.yaml"
with yaml_path.open("w", encoding="utf-8") as f:
    f.write(f"path: {YOLO_OUTPUT_DIR.resolve()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("\n")
    f.write("names:\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")


# REPORTS
split_csv = reports_dir / "split_manifest.csv"
bad_xml_csv = reports_dir / "bad_xml_or_skipped.csv"
summary_txt = reports_dir / "summary.txt"
split_distribution_csv = reports_dir / "split_object_distribution.csv"

with split_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "stem",
            "image_name",
            "label_name",
            "object_count",
            "source_image_path",
            "source_xml_path",
        ],
    )
    writer.writeheader()
    writer.writerows(all_export_rows)

with bad_xml_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["stem", "issue", "xml_path"],
    )
    writer.writeheader()
    writer.writerows(bad_xml_rows)

split_counter = Counter(row["split"] for row in all_export_rows)
split_object_totals = {
    "train": sum(r["object_count"] for r in train_records),
    "val": sum(r["object_count"] for r in val_records),
    "test": sum(r["object_count"] for r in test_records),
}

with split_distribution_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["split", "num_images", "num_objects"],
    )
    writer.writeheader()
    for split_name in ["train", "val", "test"]:
        writer.writerow({
            "split": split_name,
            "num_images": split_counter.get(split_name, 0),
            "num_objects": split_object_totals[split_name],
        })

with summary_txt.open("w", encoding="utf-8") as f:
    f.write("YOLO Dataset Prep Summary\n")
    f.write("=========================\n")
    f.write(f"Source merged dataset: {MERGED_DATASET_DIR.resolve()}\n")
    f.write(f"YOLO output folder: {YOLO_OUTPUT_DIR.resolve()}\n\n")
    f.write(f"Usable matched image/XML pairs: {len(records)}\n")
    f.write(f"Total labeled objects: {total_objects}\n")
    f.write(f"Skipped/bad XML rows: {len(bad_xml_rows)}\n\n")
    f.write("Split sizes:\n")
    f.write(f"  Train images: {len(train_records)}\n")
    f.write(f"  Val images:   {len(val_records)}\n")
    f.write(f"  Test images:  {len(test_records)}\n\n")
    f.write("Object totals by split:\n")
    f.write(f"  Train objects: {split_object_totals['train']}\n")
    f.write(f"  Val objects:   {split_object_totals['val']}\n")
    f.write(f"  Test objects:  {split_object_totals['test']}\n")

print("\nDone.")
print(f"YOLO dataset folder: {YOLO_OUTPUT_DIR}")
print(f"dataset.yaml:        {yaml_path}")
print(f"summary:             {summary_txt}")
print(f"manifest CSV:        {split_csv}")
print(f"split stats CSV:     {split_distribution_csv}")
print(f"skipped XML CSV:     {bad_xml_csv}")
