# -*- coding: utf-8 -*-
#import statements
import os
import cv2
import numpy as np
import shutil
import json
from tqdm import tqdm

#manually entering the angles the images were taken from
image_angles = {
    "IMG_8981.jpeg": "top",
    "IMG_8982.jpeg": "top",
    "IMG_9006.jpeg": "top",
    "IMG_9009.jpeg": "bottom-left",
    "IMG_9011.jpeg": "bottom-right",
    "IMG_9012.jpeg": "top-left",
    "IMG_9013.jpeg": "top-right",
    "IMG_9014.jpeg": "top-middle",
    "IMG_9015.jpeg": "bottom-middle",
    "IMG_9016.jpeg": "top",
    "IMG_9017.jpeg": "bottom-left",
    "IMG_9018.jpeg": "bottom-right",
    "IMG_9019.jpeg": "top-left",
    "IMG_9021.jpeg": "top-right",
    "IMG_9023.jpeg": "top-middle",
    "IMG_9024.jpeg": "bottom-middle",
    "IMG_9025.jpeg": "top",
    "IMG_9026.jpeg": "bottom-left",
    "IMG_9027.jpeg": "bottom-right",
    "IMG_9028.jpeg": "top-left",
    "IMG_9029.jpeg": "top-right",
    "IMG_9030.jpeg": "top-middle",
    "IMG_9031.jpeg": "bottom-middle",
    "IMG_9032.jpeg": "top",
    "IMG_9033.jpeg": "bottom-left",
    "IMG_9034.jpeg": "bottom-right",
    "IMG_9035.jpeg": "top-left",
    "IMG_9036.jpeg": "top-right",
    "IMG_9038.jpeg": "top-middle",
    "IMG_9039.jpeg": "bottom-middle",
    "IMG_9040.jpeg": "top",
    "IMG_9041.jpeg": "bottom-left",
    "IMG_9042.jpeg": "bottom-right",
    "IMG_9043.jpeg": "top-left",
    "IMG_9044.jpeg": "top-right",
    "IMG_9045.jpeg": "top-middle",
    "IMG_9046.jpeg": "bottom-middle",
    "IMG_9047.jpeg": "top",
    "IMG_9048.jpeg": "bottom-left",
    "IMG_9049.jpeg": "bottom-right",
    "IMG_9050.jpeg": "top-left",
    "IMG_9051.jpeg": "top-right",
    "IMG_9052.jpeg": "top-middle",
    "IMG_9053.jpeg": "bottom-middle"
}

#funtion to extract each zinc rock
def extract_zinc_rocks(image_path, output_dir, coords_dict, angle=None, min_area=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rock_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        single_mask = np.zeros_like(gray)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

        roi = image[y:y+h, x:x+w]
        alpha = single_mask[y:y+h, x:x+w]

        roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        roi_bgra[:, :, 3] = alpha

        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{filename_base}_rock_{rock_count}.png"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, roi_bgra)

        # Saving the coordinates and angles
        coords_dict[output_name] = {
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "image_angle": angle 
        }

        rock_count += 1

rock_coordinates = {}
output_dir = "zinc_rock_outputs"
os.makedirs(output_dir, exist_ok=True)

filenames = list(image_angles.keys())

for filename in tqdm(filenames):
    angle = image_angles.get(filename, "unknown")  
    filename = "zinc/" + filename
    extract_zinc_rocks(filename, output_dir, rock_coordinates, angle=angle)

# Saving the coordinates and angles in a JSON file
with open("zinc_rock_outputs/rock_coordinates.json", "w") as f:
    json.dump(rock_coordinates, f, indent=2)

shutil.make_archive("zinc_rock_outputs", 'zip', output_dir)