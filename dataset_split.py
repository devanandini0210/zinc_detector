#import statements
import os
import shutil
import random
from glob import glob
import yaml
from tqdm import tqdm

#extracting the input data
source_img_dir = "dataset_resized/images"
source_lbl_dir = "dataset_resized/labels"

#creating the output folders
output_base = "dataset"
output_img_dir = os.path.join(output_base, "images")
output_lbl_dir = os.path.join(output_base, "labels")

#configuring the splits and their percentages
splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]

#creating folders for each split
for split in splits:
    os.makedirs(os.path.join(output_img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_lbl_dir, split), exist_ok=True)

#gathering the data and splitting them
image_paths = sorted(glob(os.path.join(source_img_dir, "*.jpg")))
label_paths = [os.path.join(source_lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt") for p in image_paths]
pairs = [(img, lbl) for img, lbl in zip(image_paths, label_paths) if os.path.exists(lbl)]
random.shuffle(pairs)

n = len(pairs)
n_train = int(n * split_ratio[0])
n_val = int(n * split_ratio[1])
n_test = n - n_train - n_val

splits_data = {
    'train': pairs[:n_train],
    'val': pairs[n_train:n_train + n_val],
    'test': pairs[n_train + n_val:]
}

for split, items in splits_data.items():
    for img_path, lbl_path in tqdm(items):
        shutil.copy(img_path, os.path.join(output_img_dir, split, os.path.basename(img_path)))
        shutil.copy(lbl_path, os.path.join(output_lbl_dir, split, os.path.basename(lbl_path)))

#generating data.yaml file for YOLO
data_yaml = {
    'train': os.path.abspath(os.path.join(output_img_dir, 'train')),
    'val': os.path.abspath(os.path.join(output_img_dir, 'val')),
    'test': os.path.abspath(os.path.join(output_img_dir, 'test')),
    'nc': 1,
    'names': ['zinc']
}
yaml_path = os.path.join(output_base, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

splits_summary = {k: len(v) for k, v in splits_data.items()}
splits_summary