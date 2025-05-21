#import statements
import os
import json
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

#extracting the input
metadata_path = "overlayed_images/composite_metadata.json"
image_root = "overlayed_images"

#creating the output folders
output_img_dir = "augmented_dataset/images"
output_lbl_dir = "augmented_dataset/labels"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

#storing the data in COCO format 
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "zinc"}]
}
annotation_id = 1
image_id = 1

#augmentation details
transform = A.Compose([
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.9),
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

#loading the image details
with open(metadata_path, "r") as f:
    metadata = json.load(f)

#randomly selecting 80% of images to augment
image_keys = list(metadata.keys())
random.shuffle(image_keys)
subset_to_augment = set(image_keys[:int(0.8 * len(image_keys))])

#augmenting the data and saving both original and augmented image
for img_name in tqdm(image_keys, desc="Processing original and augmented"):
    entry = metadata[img_name]
    bg_path = entry["output_image"]
    image = cv2.imread(bg_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    boxes = []
    for rock in entry["rocks"]:
        x = rock["position"]["x"]
        y = rock["position"]["y"]
        bw = rock["position"]["width"]
        bh = rock["position"]["height"]
        boxes.append([x, y, x + bw, y + bh])

    labels = [0] * len(boxes)

    #saving the original image
    orig_img_path = os.path.join(output_img_dir, img_name)
    cv2.imwrite(orig_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    #saving the original YOLO label
    yolo_lines = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
    label_path = os.path.join(output_lbl_dir, img_name.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    #coco label for the original image
    coco_output["images"].append({
        "id": image_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })
    for box in boxes:
        x1, y1, x2, y2 = box
        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0
        })
        annotation_id += 1
    image_id += 1

    #augmenting the image if selected
    if img_name in subset_to_augment:
        transformed = transform(image=image, bboxes=boxes, category_ids=labels)
        aug_image = transformed["image"]
        aug_boxes = transformed["bboxes"]

        if len(aug_boxes) == 0:
            continue

        aug_name = img_name.replace(".jpg", "_aug.jpg")
        aug_img_path = os.path.join(output_img_dir, aug_name)
        cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        #YOLO label for the augmented image
        yolo_lines = []
        for box in aug_boxes:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
        aug_label_path = os.path.join(output_lbl_dir, aug_name.replace(".jpg", ".txt"))
        with open(aug_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        #COCO label for the augmented image
        coco_output["images"].append({
            "id": image_id,
            "file_name": aug_name,
            "width": w,
            "height": h
        })
        for box in aug_boxes:
            x1, y1, x2, y2 = box
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            annotation_id += 1
        image_id += 1

#saving the coco labels in a JSON file
with open("augmented_dataset/annotations_coco.json", "w") as f:
    json.dump(coco_output, f, indent=2)

#len(coco_output["images"]), len(coco_output["annotations"])