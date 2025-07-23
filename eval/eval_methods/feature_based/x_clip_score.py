from typing import List
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F
from eval_methods.utils import extract_video_frame_imgs

MAX_LENGTH = 77
MAX_NUM_FRAMES = 8


def x_clip_score_output(model_or_process, video_path: str, prompt: str) -> float:
    model=model_or_process[0]
    processor=model_or_process[1]
    tokenizer=model_or_process[2]
    device=model_or_process[3]
    
    def _read_video_frames(frame_paths: List[str], max_frames: int):
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames)).astype(int)
        selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
        return selected_frames

    frame_paths = extract_video_frame_imgs(video_path, fps=4)
    selected_frames = _read_video_frames(frame_paths, MAX_NUM_FRAMES)

    text_inputs = tokenizer([prompt], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).flatten()

        video_inputs = processor(videos=selected_frames, return_tensors="pt").to(device)
        video_features = model.get_video_features(**video_inputs).flatten()

        sim_score = F.cosine_similarity(text_features, video_features, dim=0).item()

    return float(sim_score)
