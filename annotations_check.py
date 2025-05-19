import os
import cv2
import random
from glob import glob

# === CONFIGURATION ===
image_dir = "dataset_resized/images"
label_dir = "dataset_resized/labels"
output_dir = "annotations_test"
os.makedirs(output_dir, exist_ok=True)

# === SETTINGS ===
num_samples = 10  # how many images to check
img_size = 1024   # final image size used after resizing
color = (0, 255, 0)

# === Load some random samples ===
image_paths = glob(os.path.join(image_dir, "*.jpg"))
samples = random.sample(image_paths, min(num_samples, len(image_paths)))

for img_path in samples:
    img_name = os.path.basename(img_path)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Draw boxes from YOLO labels
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            cls, x_center, y_center, w, h = map(float, line.strip().split())
            x_center *= img_size
            y_center *= img_size
            w *= img_size
            h *= img_size

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"class {int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save overlayed image
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, img)

print(f"Overlay saved in: {output_dir}")
