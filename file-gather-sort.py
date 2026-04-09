from pathlib import Path
import shutil
import csv
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

#image sources 
SOURCE_DIRS = [
    Path("/path/to/data_folder_1"),
    Path("/path/to/data_folder_2"),
    Path("/path/to/data_folder_3"),
    Path("/path/to/data_folder_4"),
    Path("/path/to/data_folder_5"),
    Path("/path/to/data_folder_6"),
    Path("/path/to/data_folder_7"),
    Path("/path/to/data_folder_8"),
    Path("/path/to/data_folder_9"),
    Path("/path/to/data_folder_10"),
]

# Final merged output folder
OUTPUT_DIR = Path("/path/to/output/merged_dataset")

# True if you want the output folder deleted and rebuilt each run
OVERWRITE_OUTPUT = True

# File types to pull recursively
IMAGE_EXTS = {".jpg", ".jpeg"}
ANNOT_EXTS = {".xml"}

# HELPER FUNCTION
def clean_source_dirs(source_dirs):
    cleaned = []
    for p in source_dirs:
        if p is None:
            continue
        p = Path(p)
        if str(p).strip() == "":
            continue
        if p.exists() and p.is_dir():
            cleaned.append(p)
        else:
            print(f"Skipping invalid folder: {p}")
    return cleaned


def reset_output_dir(output_dir, overwrite=False):
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}\n"
                f"Set OVERWRITE_OUTPUT = True to rebuild it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def find_files_recursive(folder):
    images = []
    annotations = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            images.append(p)
        elif suffix in ANNOT_EXTS:
            annotations.append(p)
    return images, annotations


def count_objects_in_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall("object"))
    except Exception as e:
        print(f"Warning: failed to parse XML {xml_path}: {e}")
        return 0


def copy_with_unique_name(src_path, dst_folder, new_name=None):
    dst_folder.mkdir(parents=True, exist_ok=True)

    if new_name is None:
        candidate = dst_folder / src_path.name
    else:
        candidate = dst_folder / new_name

    if not candidate.exists():
        shutil.copy2(src_path, candidate)
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        alt = dst_folder / f"{stem}__dup{i}{suffix}"
        if not alt.exists():
            shutil.copy2(src_path, alt)
            return alt
        i += 1


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# PREP

source_dirs = clean_source_dirs(SOURCE_DIRS)
if len(source_dirs) == 0:
    raise ValueError("No valid source folders found. Fix SOURCE_DIRS first.")

reset_output_dir(OUTPUT_DIR, overwrite=OVERWRITE_OUTPUT)

images_dir = OUTPUT_DIR / "images"
annotations_dir = OUTPUT_DIR / "annotations"
unmatched_images_dir = OUTPUT_DIR / "unmatched_images"
unmatched_annotations_dir = OUTPUT_DIR / "unmatched_annotations"
reports_dir = OUTPUT_DIR / "reports"

for d in [images_dir, annotations_dir, unmatched_images_dir, unmatched_annotations_dir, reports_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("Using source folders:")
for s in source_dirs:
    print(" -", s)

print("\nOutput folder:")
print(" -", OUTPUT_DIR)

# SCAN ALL SOURCE FOLDERS

all_images = []
all_annotations = []

for folder in source_dirs:
    imgs, anns = find_files_recursive(folder)
    all_images.extend(imgs)
    all_annotations.extend(anns)

print(f"\nFound total images: {len(all_images)}")
print(f"Found total annotations: {len(all_annotations)}")


# GROUP BY STEM

image_map = defaultdict(list)
annot_map = defaultdict(list)

for img in all_images:
    image_map[img.stem].append(img)

for ann in all_annotations:
    annot_map[ann.stem].append(ann)

all_stems = sorted(set(image_map.keys()) | set(annot_map.keys()))
matched_stems = sorted(set(image_map.keys()) & set(annot_map.keys()))
image_only_stems = sorted(set(image_map.keys()) - set(annot_map.keys()))
annot_only_stems = sorted(set(annot_map.keys()) - set(image_map.keys()))

print(f"Matched stems: {len(matched_stems)}")
print(f"Image-only stems: {len(image_only_stems)}")
print(f"Annotation-only stems: {len(annot_only_stems)}")

# HANDLE MATCHED FILES

per_image_rows = []
duplicate_rows = []
distribution_counts = []

for stem in matched_stems:
    img_list = sorted(image_map[stem], key=lambda p: str(p))
    ann_list = sorted(annot_map[stem], key=lambda p: str(p))

    # Keep first image and first xml as the primary pair
    primary_img = img_list[0]
    primary_ann = ann_list[0]

    # Record duplicates if more than one image or xml exists for same stem
    if len(img_list) > 1:
        for extra in img_list[1:]:
            duplicate_rows.append({
                "stem": stem,
                "file_type": "image",
                "kept_file": str(primary_img),
                "duplicate_file": str(extra),
            })

    if len(ann_list) > 1:
        for extra in ann_list[1:]:
            duplicate_rows.append({
                "stem": stem,
                "file_type": "annotation",
                "kept_file": str(primary_ann),
                "duplicate_file": str(extra),
            })

    copied_img = copy_with_unique_name(primary_img, images_dir, new_name=primary_img.name)
    copied_ann = copy_with_unique_name(primary_ann, annotations_dir, new_name=primary_ann.name)

    obj_count = count_objects_in_xml(primary_ann)
    distribution_counts.append(obj_count)

    per_image_rows.append({
        "stem": stem,
        "image_name": copied_img.name,
        "annotation_name": copied_ann.name,
        "object_count": obj_count,
        "source_image_path": str(primary_img),
        "source_annotation_path": str(primary_ann),
    })

# HANDLE UNMATCHED FILES


unmatched_rows = []

for stem in image_only_stems:
    for img in sorted(image_map[stem], key=lambda p: str(p)):
        copied = copy_with_unique_name(img, unmatched_images_dir, new_name=img.name)
        unmatched_rows.append({
            "stem": stem,
            "file_type": "image_only",
            "copied_name": copied.name,
            "source_path": str(img),
        })

for stem in annot_only_stems:
    for ann in sorted(annot_map[stem], key=lambda p: str(p)):
        copied = copy_with_unique_name(ann, unmatched_annotations_dir, new_name=ann.name)
        unmatched_rows.append({
            "stem": stem,
            "file_type": "annotation_only",
            "copied_name": copied.name,
            "source_path": str(ann),
        })

# BUILD DISTRIBUTION TABLE

distribution_counter = Counter(distribution_counts)
distribution_rows = []

for obj_count in sorted(distribution_counter):
    distribution_rows.append({
        "objects_in_image": obj_count,
        "number_of_images": distribution_counter[obj_count]
    })

# SAVE

per_image_csv = reports_dir / "per_image_object_counts.csv"
distribution_csv = reports_dir / "object_count_distribution.csv"
duplicates_csv = reports_dir / "duplicates_report.csv"
unmatched_csv = reports_dir / "unmatched_files_report.csv"
summary_txt = reports_dir / "summary.txt"
plot_png = reports_dir / "object_count_distribution.png"

write_csv(
    per_image_csv,
    fieldnames=[
        "stem",
        "image_name",
        "annotation_name",
        "object_count",
        "source_image_path",
        "source_annotation_path",
    ],
    rows=per_image_rows
)

write_csv(
    distribution_csv,
    fieldnames=["objects_in_image", "number_of_images"],
    rows=distribution_rows
)

write_csv(
    duplicates_csv,
    fieldnames=["stem", "file_type", "kept_file", "duplicate_file"],
    rows=duplicate_rows
)

write_csv(
    unmatched_csv,
    fieldnames=["stem", "file_type", "copied_name", "source_path"],
    rows=unmatched_rows
)

total_objects = sum(distribution_counts)
num_matched = len(per_image_rows)
min_objects = min(distribution_counts) if distribution_counts else None
max_objects = max(distribution_counts) if distribution_counts else None
mean_objects = (total_objects / num_matched) if num_matched > 0 else None

with summary_txt.open("w", encoding="utf-8") as f:
    f.write("Merged Dataset Summary\n")
    f.write("======================\n")
    f.write(f"Number of source folders used: {len(source_dirs)}\n")
    f.write(f"Total image files found recursively: {len(all_images)}\n")
    f.write(f"Total annotation files found recursively: {len(all_annotations)}\n")
    f.write(f"Matched image/xml pairs kept: {num_matched}\n")
    f.write(f"Image-only stems: {len(image_only_stems)}\n")
    f.write(f"Annotation-only stems: {len(annot_only_stems)}\n")
    f.write(f"Duplicate file rows recorded: {len(duplicate_rows)}\n")
    f.write(f"Total labeled objects across kept pairs: {total_objects}\n")
    f.write(f"Min objects in an image: {min_objects}\n")
    f.write(f"Max objects in an image: {max_objects}\n")
    f.write(f"Mean objects per image: {mean_objects}\n")

# PLOT DISTRIBUTION

x_vals = sorted(distribution_counter.keys())
y_vals = [distribution_counter[x] for x in x_vals]

plt.figure(figsize=(10, 6))
plt.bar(x_vals, y_vals)
plt.xlabel("Number of objects in an image")
plt.ylabel("Number of images")
plt.title("Distribution of Object Counts per Image")
plt.tight_layout()
plt.savefig(plot_png, dpi=200)
plt.show()


# FINAL PRINTS


print("\nDone.")
print(f"Merged images folder:        {images_dir}")
print(f"Merged annotations folder:   {annotations_dir}")
print(f"Unmatched images folder:     {unmatched_images_dir}")
print(f"Unmatched annotations folder:{unmatched_annotations_dir}")
print(f"Per-image report:            {per_image_csv}")
print(f"Distribution report:         {distribution_csv}")
print(f"Duplicates report:           {duplicates_csv}")
print(f"Unmatched report:            {unmatched_csv}")
print(f"Summary:                     {summary_txt}")
print(f"Plot:                        {plot_png}")
