import os
import json
import shutil
from PIL import Image
from src.utils.yolo_to_coco import yolo_to_coco

def prepare_coco_dataset():
    # Input paths
    original_base = "data/12_RGB_FullyLabeled_640/coco"
    classes_txt = os.path.join(original_base, "classes.txt")

    # Output structure
    dataset_dir = "dataset"
    os.makedirs(os.path.join(dataset_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)

    # Copy and convert train
    print("Processing train set...")
    train_img_in = os.path.join(original_base, "train/images")
    train_lbl = os.path.join(original_base, "train/labels")
    train_img_out = os.path.join(dataset_dir, "images/train")
    for f in os.listdir(train_img_in):
        shutil.copy(os.path.join(train_img_in, f), train_img_out)
    train_json = os.path.join(dataset_dir, "annotations/instances_train.json")
    next_img_id, next_ann_id = yolo_to_coco(train_img_out, train_lbl, classes_txt, train_json)

    # Copy and convert val
    print("Processing val set...")
    val_img_in = os.path.join(original_base, "val/images")
    val_lbl = os.path.join(original_base, "val/labels")
    val_img_out = os.path.join(dataset_dir, "images/val")
    for f in os.listdir(val_img_in):
        shutil.copy(os.path.join(val_img_in, f), val_img_out)
    val_json = os.path.join(dataset_dir, "annotations/instances_val.json")
    yolo_to_coco(val_img_out, val_lbl, classes_txt, val_json, start_image_id=next_img_id, start_annotation_id=next_ann_id)

    print("✅ COCO-format dataset prepared at ./dataset")


if __name__ == "__main__":
    prepare_coco_dataset()