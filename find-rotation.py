import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_DIR = Path("images")
XML_DIR = Path("annotations")

OUT_IMG_DIR = Path("fixed/images")
OUT_XML_DIR = Path("fixed/annotations")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_XML_DIR.mkdir(parents=True, exist_ok=True)

ROTATION = 90  # use 90 or 270

def rotate_box_90cw(xmin, ymin, xmax, ymax, w, h):
    return h - ymax, xmin, h - ymin, xmax

def rotate_box_270cw(xmin, ymin, xmax, ymax, w, h):
    return ymin, w - xmax, ymax, w - xmin

for xml_path in XML_DIR.glob("*.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.findall("object"):
        b = obj.find("bndbox")

        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))

        if ROTATION == 90:
            nxmin, nymin, nxmax, nymax = rotate_box_90cw(xmin, ymin, xmax, ymax, w, h)
        elif ROTATION == 270:
            nxmin, nymin, nxmax, nymax = rotate_box_270cw(xmin, ymin, xmax, ymax, w, h)
        else:
            raise ValueError("ROTATION must be 90 or 270")

        b.find("xmin").text = str(max(0, nxmin))
        b.find("ymin").text = str(max(0, nymin))
        b.find("xmax").text = str(min(w, nxmax))
        b.find("ymax").text = str(min(h, nymax))

    tree.write(OUT_XML_DIR / xml_path.name)

    # copy matching image
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = IMG_DIR / f"{xml_path.stem}{ext}"
        if img_path.exists():
            shutil.copy(img_path, OUT_IMG_DIR / img_path.name)
            break

print("Fixed XMLs saved to:", OUT_XML_DIR)
