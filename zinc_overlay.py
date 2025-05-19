import os
import json
import cv2
import random
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
rock_json = "zinc_rock_outputs/rock_coordinates.json"
rock_png_folder = "zinc_rock_outputs/"
background_root = "output_frames/"
output_root = "overlayed_images/"
os.makedirs(output_root, exist_ok=True)

# === IMAGE RESOLUTION USED FOR ROCK EXTRACTION ===
src_width = 4032
src_height = 3024

# === LOAD ROCK METADATA ===
with open(rock_json, "r") as f:
    rock_data = json.load(f)

# === GROUP ROCKS BY ANGLE ===
rocks_by_angle = {}
for rock_file, meta in rock_data.items():
    angle = meta["image_angle"]
    bbox = meta["bounding_box"]
    rocks_by_angle.setdefault(angle, []).append({
        "filename": rock_file,
        "bbox": bbox
    })

# === OUTPUT METADATA CONTAINER ===
composite_metadata = {}
image_counter = 0

# === PROCESS EACH ANGLE FOLDER ===
for angle_folder in os.listdir(background_root):
    full_angle_path = os.path.join(background_root, angle_folder)
    if not os.path.isdir(full_angle_path):
        continue

    angle_key = angle_folder.replace("frames_", "")
    rocks = rocks_by_angle.get(angle_key, [])
    if not rocks:
        print(f"No rocks for angle: {angle_key}")
        continue

    bg_images = glob(os.path.join(full_angle_path, "*.jpg"))
    if not bg_images:
        print(f"No images found in {full_angle_path}")
        continue

    output_angle_folder = os.path.join(output_root, angle_key)
    os.makedirs(output_angle_folder, exist_ok=True)

    for bg_path in tqdm(bg_images, desc=f"Processing {angle_key}"):
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        h, w = bg.shape[:2]

        image_name = f"composited_{image_counter:04}.jpg"
        output_path = os.path.join(output_angle_folder, image_name)

        rocks_on_image = []
        num_rocks = random.randint(5, 20)
        chosen_rocks = random.choices(rocks, k=num_rocks)

        for rock in chosen_rocks:
            bbox = rock["bbox"]
            rock_path = os.path.join(rock_png_folder, rock["filename"])
            rock_img = cv2.imread(rock_path, cv2.IMREAD_UNCHANGED)
            if rock_img is None:
                continue

            w_box, h_box = bbox["width"], bbox["height"]
            cx_src = bbox["x"] + w_box // 2
            cy_src = bbox["y"] + h_box // 2
            rel_x = cx_src / src_width
            rel_y = cy_src / src_height
            cx_new = int(rel_x * w)
            cy_new = int(rel_y * h)
            x = cx_new - w_box // 2
            y = cy_new - h_box // 2

            # Ensure rock fits inside new background
            if x < 0 or y < 0 or x + w_box > w or y + h_box > h:
                continue

            # Paste rock with alpha
            overlay = rock_img[:, :, :3]
            alpha = rock_img[:, :, 3:] / 255.0
            roi = bg[y:y+h_box, x:x+w_box]
            blended = (1 - alpha) * roi + alpha * overlay
            bg[y:y+h_box, x:x+w_box] = blended.astype("uint8")

            rocks_on_image.append({
                "rock_file": rock["filename"],
                "position": {
                    "x": x,
                    "y": y,
                    "width": w_box,
                    "height": h_box
                }
            })

        # Save final image
        cv2.imwrite(output_path, bg)

        # Store metadata
        composite_metadata[image_name] = {
            "background_image": bg_path,
            "output_image": output_path,
            "angle": angle_key,
            "rocks": rocks_on_image
        }

        image_counter += 1

# === SAVE METADATA ===
with open(os.path.join(output_root, "composite_metadata.json"), "w") as f:
    json.dump(composite_metadata, f, indent=2)

print(f"\n Overlay complete. Saved {image_counter} composited images.")
