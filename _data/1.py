import sys
sys.path.append('D:\python\generative-image-dynamics')

from tqdm import tqdm
import csv
import os
import cv2
import numpy as np
from pathlib import Path

from utils import *
from utils.flow import optical_flow

from dataclasses import dataclass

@dataclass
class VideoData:
    video_path: Path
    flow_path: list[Path]
    label_path: Path


dataset_folder = Path("./_data")
video_folder = dataset_folder / "video"
flow_folder = dataset_folder / "flow"
label_fodler = dataset_folder / "label"

video_folder.mkdir(parents=True, exist_ok=True)
flow_folder.mkdir(parents=True, exist_ok=True)
label_fodler.mkdir(parents=True, exist_ok=True)

csv_path = label_fodler /  "motion_synthesis_train_set.csv"

num_frames = 75
width = 288
height = 288
train_set = []
train_set_ids = []



def find_video_files(directory):
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file[0] == ".":
        continue
      if file.lower().endswith('.mp4'):
        yield os.path.join(root, file)

def parse_one_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    videoid = os.path.basename(video_file).replace(".mp4", "")
    print(f">> video: {videoid} fps: {fps} frames: {frame_count}")
    
    if fps == 29 or fps == 23:
        fps += 1

    seq_count = frame_count // num_frames
    if seq_count == 0:
        return None, None, None
    offset = 0
    seqs = [int((ii * num_frames + offset) / fps) for ii in range(seq_count)]
    return videoid, fps, seqs

def write_csv():
    fields = ['video_id', 'start_sec', 'num_frames', 'frames_per_sec']
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_set)

def load_csv(checkvideo_exists=False):
    if not csv_path.exists():
        return
    missing_videoids = []

    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            videoid = row["video_id"]
            time_s = int(row["start_sec"])
            if checkvideo_exists:
                if videoid in missing_videoids:
                    continue
                videofile = os.path.join(video_folder, videoid + ".mp4")
                if not os.path.exists(videofile):
                    print("found missing video:", videoid)
                    missing_videoids.append(videoid)
                    continue

            flowfile = os.path.join(flow_folder,f"{videoid}_{time_s:03d}.npy")
            if not os.path.exists(flowfile):
                continue
            if not os.path.exists(flowfile):
                continue
            train_set.append(row)
            
            if videoid not in train_set_ids:
                train_set_ids.append(videoid)

def fix_missing_video():
    load_csv(checkvideo_exists=True)
    print("total items:", len(train_set))
    write_csv()
    print("--- done!")


load_csv()
print("previous records:", len(train_set))
print("skped:", len(train_set_ids), train_set_ids)

all_videos = list(find_video_files(video_folder))
all_videos.sort()
print("all_videos:", len(all_videos))

for video in tqdm(all_videos):
    videoid, fps, seqs = parse_one_video(video)
    if videoid is None:
        continue
    if videoid in train_set_ids:
        continue
    for time_s in seqs:
        frames = get_frames(video, w=width, h=height, start_sec=time_s, fps=fps, f=num_frames)
        if frames is None or len(frames) == 0:
            print("!! video failed to load", videoid)
            continue
            
        flow_path = os.path.join(flow_folder,f"{videoid}_{time_s:03d}.npy")
        if not os.path.exists(flow_path):
            flow = optical_flow(frames[0], frames[1:])
            save_npy(flow, flow_path, dtype=np.float16)
            
        train_set.append(dict(video_id=videoid, start_sec=time_s, num_frames=num_frames, frames_per_sec=fps))
    train_set_ids.append(videoid)
print("write csv!")
write_csv()


