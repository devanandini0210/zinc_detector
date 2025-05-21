#import statements
import cv2
import os
from tqdm import tqdm

#accesing the folder contining the videos of non-zinc materials
video_dir = 'non-zinc'

#creating output folder
output_base = 'output_frames'
os.makedirs(output_base, exist_ok=True)

#manually initialising the number of frames to skip after each extraction based on the speed of the video
frame_per_video = {
    "IMG_9066.MOV": 10,
    "IMG_9072.MOV": 15,
    "IMG_9073.MOV": 10,
    "IMG_9067.MOV": 10,
    "IMG_9059.MOV": 15,
    "IMG_9071.MOV": 15,
    "IMG_9065.MOV": 15,
    "IMG_9070.MOV": 10,
    "IMG_9058.MOV": 10,
    "IMG_9074.MOV": 10,
    "IMG_9061.MOV": 10,
    "IMG_9062.MOV": 15,
    "IMG_9068.MOV": 7,
    "IMG_9056.MOV": 20,
}

#manually initialising the angles the videos were taken from, same as in zinc_extractor.py
video_angle = {
    "IMG_9066.MOV": 'bottom-middle',
    "IMG_9072.MOV": 'top-right',
    "IMG_9073.MOV": 'top-middle',
    "IMG_9067.MOV": 'top',
    "IMG_9059.MOV": 'bottom-right',
    "IMG_9071.MOV": 'top-left',
    "IMG_9065.MOV": 'top-middle',
    "IMG_9070.MOV": 'bottom-right',
    "IMG_9058.MOV": 'bottom-left',
    "IMG_9074.MOV": 'bottom-middle',
    "IMG_9061.MOV": 'top-left',
    "IMG_9062.MOV": 'top-right',
    "IMG_9068.MOV": 'bottom-left',
    "IMG_9056.MOV": 'top',
}

#initialising supported video formats
video_extensions = ('.mp4', '.mov', '.avi', '.mkv')

#adding the video files to a list
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
video_files.sort()

#extracting the frames from each video
for video_file in tqdm(video_files):
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_base, f'{video_angle[video_file]}')

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = len(os.listdir(output_folder))

    frame_factor = frame_per_video[video_file]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_factor == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_idx:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    
print("Frame Extraction Completed")