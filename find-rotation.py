import random
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt

# CHANGE THESE
IMG_DIR = Path("images")
XML_DIR = Path("annotations")

NUM_SAMPLES = 4

def read_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        b = obj.find("bndbox")

        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))

        boxes.append((name, xmin, ymin, xmax, ymax))

    return boxes

def rotate_box_90cw(box, w, h):
    name, xmin, ymin, xmax, ymax = box
    return name, h - ymax, xmin, h - ymin, xmax

def rotate_box_270cw(box, w, h):
    name, xmin, ymin, xmax, ymax = box
    return name, ymin, w - xmax, ymax, w - xmin

def draw_boxes(img, boxes):
    img = img.copy()

    for name, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(
            img,
            name,
            (xmin, max(20, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return img

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(list(IMG_DIR.glob(ext)))

valid_pairs = []
for img_path in image_paths:
    xml_path = XML_DIR / f"{img_path.stem}.xml"
    if xml_path.exists():
        valid_pairs.append((img_path, xml_path))

samples = random.sample(valid_pairs, min(NUM_SAMPLES, len(valid_pairs)))

for img_path, xml_path in samples:
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    boxes = read_boxes(xml_path)

    original = draw_boxes(img, boxes)
    boxes_90 = [rotate_box_90cw(b, w, h) for b in boxes]
    boxes_270 = [rotate_box_270cw(b, w, h) for b in boxes]

    img_90 = draw_boxes(img, boxes_90)
    img_270 = draw_boxes(img, boxes_270)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title(f"{img_path.name}\nOriginal")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_90)
    plt.title("Boxes rotated 90° CW")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_270)
    plt.title("Boxes rotated 270° CW")
    plt.axis("off")

    plt.show()
