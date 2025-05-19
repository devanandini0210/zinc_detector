import cv2
import numpy as np
import os
import json
import random
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
rocks_folder = "zinc_rock_outputs/"  # folder with transparent PNG rocks
metadata_file = "zinc_rock_outputs/rock_coordinates.json"  # metadata from extraction step
background_root = "output_frames/"  # root folder with subfolders per angle
output_folder = "overlayed_images"
os.makedirs(output_folder, exist_ok=True)

ellipse_width_ratio = 0.6
ellipse_height_ratio = 0.6

# === LOAD ROCK METADATA ===
with open(metadata_file, "r") as f:
    rock_data = json.load(f)

composite_metadata = {}


# Organize rocks by angle
dict_by_angle = {}
for rock_file, meta in rock_data.items():
    angle = meta["image_angle"]
    dict_by_angle.setdefault(angle, []).append({
        "filename": rock_file,
        "bbox": meta["bounding_box"]
    })

# === PROCESS EACH ANGLE FOLDER ===
for angle_folder in os.listdir(background_root):
    full_angle_path = os.path.join(background_root, angle_folder)
    if not os.path.isdir(full_angle_path):
        continue

    angle_folder = angle_folder.replace("frames_", "")
    rocks = dict_by_angle.get(angle_folder, [])


    rocks = dict_by_angle.get(angle_folder, [])
    if not rocks:
        print(f"No rocks for angle: {angle_folder}")
        continue

    background_paths = glob(os.path.join(full_angle_path, "*.jpg"))

    for bg_path in tqdm(background_paths):
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        h, w = bg.shape[:2]

        # Save output
        out_name = os.path.splitext(os.path.basename(bg_path))[0] + "_composited.png"
        # Create subfolder based on angle
        angle_output_folder = os.path.join(output_folder, angle_folder)
        os.makedirs(angle_output_folder, exist_ok=True)
        # Init metadata record
        composite_metadata[out_name] = {
            "background_image": bg_path,
            "output_image": os.path.join(output_folder, angle_folder, out_name),
            "angle": angle_folder,
            "rocks": []
        }
        


        # Define ellipse area
        ellipse_center = (w // 2, h // 2)
        ellipse_axes = (int(w * ellipse_width_ratio / 2), int(h * ellipse_height_ratio / 2))

        # Create mask for ellipse
        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

        # Select N rocks
        num_rocks = random.randint(5, 20)
        placed_count = 0
        max_attempts = 100  # avoid infinite loop

        attempted_rocks = set()

        while placed_count < num_rocks and len(attempted_rocks) < max_attempts:
            rock_info = random.choice(rocks)
            rock_id = rock_info["filename"]

            if rock_id in attempted_rocks:
                continue
            attempted_rocks.add(rock_id)

            rock_path = os.path.join(rocks_folder, rock_info["filename"])
            bbox = rock_info["bbox"]
            rock_img = cv2.imread(rock_path, cv2.IMREAD_UNCHANGED)
            if rock_img is None:
                continue
            
            bbox = rock_info["bbox"]
            w_box, h_box = bbox["width"], bbox["height"]    
            # === Use relative position from original image ===
            cx_src = bbox["x"] + w_box // 2
            cy_src = bbox["y"] + h_box // 2

            # Manually set your source image resolution if known
            # (e.g., 1024x768 or the resolution you used to extract rocks)
            src_width = 4032
            src_height = 3024


            # Map to new image dimensions
            rel_x = cx_src / src_width
            rel_y = cy_src / src_height

            cx_new = int(rel_x * w)
            cy_new = int(rel_y * h)

            # Top-left corner of the box
            x = cx_new - w_box // 2
            y = cy_new - h_box // 2
            #print(f"Rock: {rock_info['filename']}")
            #print(f"Original center: ({cx_src}, {cy_src}), New center: ({cx_new}, {cy_new})")
            #print(f"New top-left corner: ({x}, {y}), box size: ({w_box}, {h_box})")
            # Make sure it fits
            if x < 0 or y < 0 or x + w_box > w or y + h_box > h:
                continue
            
            # Must lie inside the ellipse
            if ellipse_mask[cy_new, cx_new] == 0:
                continue

            if x < 0 or y < 0 or x + w_box > w or y + h_box > h:
                print("Skipping: out of bounds.")
                continue
            
            if ellipse_mask[cy_new, cx_new] == 0:
                print("Skipping: outside ellipse.")
                continue


            # Paste RGBA onto BGR
            overlay = rock_img[:, :, :3]
            mask = rock_img[:, :, 3:] / 255.0
            roi = bg[y:y+h_box, x:x+w_box]
            blended = (1 - mask) * roi + mask * overlay
            bg[y:y+h_box, x:x+w_box] = blended.astype(np.uint8)

            
            # Collect overlay metadata
            composite_metadata.setdefault(out_name, {
                "background_image": bg_path,
                "output_image": os.path.join(angle_output_folder, out_name),
                "angle": angle_folder,
                "rocks": []
            })["rocks"].append({
                "rock_file": rock_info["filename"],
                "position": {
                    "x": x,
                    "y": y,
                    "width": w_box,
                    "height": h_box
                }
            })

            
            placed_count += 1



        

        
        out_path = os.path.join(angle_output_folder, out_name)
        cv2.imwrite(out_path, bg)
        # print(f"Saved: {out_path}")


 # Save composite metadata

with open(os.path.join(output_folder, "composite_metadata.json"), "w") as f:
    json.dump(composite_metadata, f, indent=2)


