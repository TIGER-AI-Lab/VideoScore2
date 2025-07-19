import base64
import cv2
from typing import List
import os
from tqdm import tqdm
import requests


def extract_video_frames_base64(video_path: str) -> List[str]:
    MAX_FRAMES=64
    SAMPLE_NUM_LOW=8
    SAMPLE_NUM_HIGH=12
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames <= MAX_FRAMES:
        target_indices = list(range(min(SAMPLE_NUM_LOW, total_frames)))
    else:
        step = total_frames / SAMPLE_NUM_HIGH
        target_indices = [int(i * step) for i in range(12)]

    current_frame = 0
    target_set = set(target_indices)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in target_set:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames.append(frame_b64)
            if len(frames) >= len(target_indices):
                break
        current_frame += 1

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