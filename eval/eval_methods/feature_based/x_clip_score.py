from typing import List
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F
from eval_methods.utils import extract_video_frame_imgs

MAX_LENGTH = 77
MAX_NUM_FRAMES = 8

X_CLIP_POINT_1=0.15
X_CLIP_POINT_2=0.2
X_CLIP_POINT_3=0.25
X_CLIP_POINT_4=0.30

def x_clip_score_output(model_or_process, video_path: str, prompt: str) -> float:
    model=model_or_process[0]
    processor=model_or_process[1]
    tokenizer=model_or_process[2]
    device=model.device
    
    frame_paths= extract_video_frame_imgs(video_path,)
    
    def _read_video_frames(frame_paths, max_frames):
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)

        selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
        return np.stack(selected_frames)
    
    input_text = tokenizer([prompt], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt").to(device)
    text_feature = model.get_text_features(**input_text).flatten()

    video=_read_video_frames(frame_paths,MAX_NUM_FRAMES)
    
    input_video = processor(videos=list(video), return_tensors="pt").to(device)
    video_feature = model.get_video_features(**input_video).flatten()
    score=F.cosine_similarity(text_feature, video_feature, dim=0).item()
    if score < X_CLIP_POINT_1:
        score = 1
    elif score >= X_CLIP_POINT_1 and score < X_CLIP_POINT_2:
        score = 2
    elif score >= X_CLIP_POINT_2 and score < X_CLIP_POINT_3:
        score = 3
    elif score >= X_CLIP_POINT_3 and score < X_CLIP_POINT_4:
        score = 4
    else:
        score=5
    return score
