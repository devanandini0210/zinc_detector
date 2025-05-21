#import statements
import os
import cv2
import json
from glob import glob
from tqdm import tqdm

#extracting the input data
input_img_dir = "augmented_dataset/images"
input_lbl_dir = "augmented_dataset/labels"
input_coco_json = "augmented_dataset/annotations_coco.json"

#creating the output folders
output_img_dir = "dataset_resized/images"
output_lbl_dir = "dataset_resized/labels"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

#initialising the target size for the resizing
target_size = 1024
resized_image_map = {} 

image_paths = sorted(glob(os.path.join(input_img_dir, "*.jpg")))
print(f"Resizing {len(image_paths)} images to {target_size}x{target_size}...")

#resizing the images and updating the YOLO labels
for img_path in tqdm(image_paths, desc="Resizing images"):
    #resizing the images
    img = cv2.imread(img_path)
    h_old, w_old = img.shape[:2]
    img_resized = cv2.resize(img, (target_size, target_size))
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_img_dir, img_name), img_resized)
    resized_image_map[img_name] = (w_old, h_old)

    #updating the YOLO labels
    label_path = os.path.join(input_lbl_dir, img_name.replace(".jpg", ".txt"))
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()
        with open(os.path.join(output_lbl_dir, img_name.replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(lines))

#updating the COCO annotations
print("Updating COCO annotations...")
with open(input_coco_json, "r") as f:
    coco_data = json.load(f)

for img in coco_data["images"]:
    filename = img["file_name"]
    if filename in resized_image_map:
        img["width"] = target_size
        img["height"] = target_size

scale_factors = {
    name: (target_size / w, target_size / h)
    for name, (w, h) in resized_image_map.items()
}

for ann in tqdm(coco_data["annotations"], desc="Adjusting COCO boxes"):
    img_id = ann["image_id"]
    image_info = next((img for img in coco_data["images"] if img["id"] == img_id), None)
    if not image_info:
        continue
    filename = image_info["file_name"]
    if filename not in scale_factors:
        continue
    sx, sy = scale_factors[filename]
    x, y, w, h = ann["bbox"]
    ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]
    ann["area"] = ann["bbox"][2] * ann["bbox"][3]

#saving the resized COCO annotations
output_coco_path = "dataset_resized/annotations_coco.json"
with open(output_coco_path, "w") as f:
    json.dump(coco_data, f, indent=2)