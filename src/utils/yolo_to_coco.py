import os
import json
from PIL import Image
import pandas as pd

# def load_classes(txt_path):
#     with open(txt_path, "r") as f:
#         return [line.strip() for line in f.readlines()]

def load_classes_from_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)
    return df[0].tolist()

def convert_yolo_to_coco(img_dir, label_dir, classes):
    images = []
    annotations = []
    ann_id = 1
    image_id = 0

    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # Load image size
        with Image.open(img_path) as img:
            width, height = img.size

        image_id += 1
        images.append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # Read YOLO annotation
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, xc, yc, w, h = map(float, parts)
                class_id = int(class_id)

                # Convert to COCO bbox format
                x = (xc - w / 2) * width
                y = (yc - h / 2) * height
                w *= width
                h *= height

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

    categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(classes)
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

# === Paths ===
base_path = "data/12_RGB_FullyLabeled_640/coco"
xlsx_path = os.path.join(base_path, "class12_RGB_all_L.xlsx")
classes = load_classes_from_xlsx(xlsx_path)

splits = ['train', 'val']
for split in splits:
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")
    output = os.path.join(base_path, "annotations", f"instances_{split}.json")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    coco_data = convert_yolo_to_coco(img_dir, label_dir, classes)

    with open(output, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"[✔] Saved COCO annotations: {output}")
