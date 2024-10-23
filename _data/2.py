import cv2

from loguru import logger
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path


class SubVideoData:

    def __init__(self, video_path: Path, start_time: int, num_frame: int, fps: int) -> None:
        self.start_time = start_time
        self.num_frame = num_frame
        self.fps = fps

        self.video_name: str = video_path.stem

        self.video_path: Path = video_path
        self.flow_path: Path = video_path.parent / f"flow/{self.video_name}__{start_time}_{num_frame}_{fps}.npy"

    def check_flow():
        pass


# def process_video(video_path):
#     print(f"开始处理视频：{video_path}")
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return len(frames)

if __name__ == "__main__":
    dataset_folder = Path("./_data")
    video_folder = dataset_folder / "video"
    flow_folder = dataset_folder / "flow"

    flow_folder.mkdir(parents=True, exist_ok=True)

    num_frames = 75
    width = 288
    height = 288
    train_set = []
    train_set_ids = []

    for video_path in video_folder.glob("*.mp4"):
        video_cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        video_num_frame: int = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"处理: {video_path}  frame_num: {video_num_frame}  fps: {video_fps}")
        
        video_ret:bool = True
        frame: cv2.Mat
        frame_list: list[cv2.Mat] = []
        while video_ret:
            video_ret, frame = video_cap.read()
            if video_ret:
                frame_list.append(frame)
        logger.info("\t视频读取完毕")

        for start_time in range(video_num_frame // num_frames):
            sub_video_data = SubVideoData(video_path=video_path, start_time=start_time, num_frame=num_frames, fps=video_fps)
            logger.info(f"\t子视频: start: {start_time}  num_frame: {num_frames}  fps: {video_fps}")

            if sub_video_data.check_flow():
                continue
                
            
