import cv2
import imageio
import pyflow
import subprocess
import multiprocessing

import numpy as np

from pathlib import Path
from loguru import logger


def optical_flow(src, tgt):
    """Optical flow from the source frame to each target frame using pyflow (https://github.com/pathak22/pyflow)."""
    para = dict(alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=0)

    assert src.dtype == np.uint8 and len(src.shape) == 3, src.shape
    assert tgt.dtype == np.uint8 and len(tgt.shape) in [3, 4], tgt.shape
    assert tgt.shape[-3:] == src.shape, (src.shape, tgt.shape)

    src = src.astype(float) / 255
    tgt = tgt.astype(float) / 255

    if len(tgt.shape) == 3:
        *uv, _ = pyflow.coarse2fine_flow(src, tgt, **para)
        return np.stack(uv, axis=2)
    else:
        flow = []
        for im in tgt:
            *uv, _ = pyflow.coarse2fine_flow(src, im, **para)
            flow.append(np.stack(uv, axis=2))
        return np.stack(flow)


def get_frames(inp: str, w: int, h: int, start_sec: float = 0, t: float = None, f: int = None, fps=None) -> np.ndarray:
    args = []
    if t is not None:
        args += ["-t", f"{t:.2f}"]
    elif f is not None:
        args += ["-frames:v", str(f)]
    if fps is not None:
        args += ["-r", str(fps)]

    args = [
        "ffmpeg", "-nostdin", "-ss", f"{start_sec:.2f}", "-i", inp, *args, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
        f"{w}x{h}", "pipe:"
    ]

    process = subprocess.Popen(args, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{inp}: ffmpeg error: {err.decode('utf-8')}")

    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def work_fun(video_file: Path, width: int, height: int, sub_video_start_time: float, video_fps: int, num_frames: int,
             sub_frame_idx: int, sub_video_file: Path, sub_video_label_file: Path):

    frames = get_frames(video_file, w=width, h=height, start_sec=sub_video_start_time, fps=video_fps, f=num_frames)
    logger.info(f'{video_file} {sub_frame_idx} {frames.shape}')

    with imageio.get_writer(sub_video_file, fps=video_fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    flow = optical_flow(frames[0], frames[1:])
    np.savez_compressed(sub_video_label_file, flow.astype(np.float16))


if __name__ == "__main__":
    dataset_folder = Path("_data")

    video_folder = dataset_folder / "video"
    sub_video_folder = dataset_folder / "sub_video"
    sub_video_label_fodler = dataset_folder / "sub_video_label"

    sub_video_folder.mkdir(parents=True, exist_ok=True)
    sub_video_label_fodler.mkdir(parents=True, exist_ok=True)

    num_frames = 75
    width = 288
    height = 288

    video_file_list = sorted(list(video_folder.glob('*.mp4')), key=lambda x: x.name)

    with multiprocessing.Pool(2) as pool:
        for video_file in video_file_list:
            cap = cv2.VideoCapture(video_file)
            video_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if video_fps in [23, 29]:
                video_fps += 1

            for sub_frame_idx in range(video_frame_num // num_frames):
                sub_video_start_time = sub_frame_idx * (num_frames / video_fps)
                sub_video_file = sub_video_folder / f'{video_file.stem}_{sub_video_start_time}_{num_frames}_{video_fps}.mp4'
                sub_video_label_file = sub_video_label_fodler / f'{video_file.stem}_{sub_video_start_time}_{num_frames}_{video_fps}.npz'

                if sub_video_label_file.exists():
                    logger.info(f'{sub_video_label_file.name} exit\tcontinue')
                    continue

                pool.apply_async(work_fun, (video_file, width, height, sub_video_start_time, video_fps, num_frames, sub_frame_idx,
                                            sub_video_file, sub_video_label_file))

        pool.close()
        pool.join()
