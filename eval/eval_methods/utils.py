import base64
import cv2
from typing import List
import os
from tqdm import tqdm
import requests


def extract_video_frames_base64(video_path: str, fps: float = 2.0) -> List[str]:
    MAX_FRAMES = 64

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    target_num = min(int(duration * fps), MAX_FRAMES)

    if target_num == 0:
        raise ValueError("Too few frames to sample from the video.")

    step = max(int(total_frames / target_num), 1)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        frames.append(frame_b64)
        if len(frames) >= target_num:
            break

    cap.release()
    return frames


def _download_file(url: str, save_path: str, overwrite: bool = False, timeout: int = 15):
    chunk_size=1<<14
    if os.path.exists(save_path) and not overwrite:
        print(f"[skip] {save_path} already exists")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    print(f"[ok] Downloaded → {save_path}")