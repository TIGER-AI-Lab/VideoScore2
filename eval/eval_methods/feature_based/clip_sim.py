import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
import numpy as np

from eval_methods.utils import extract_video_frame_imgs

CLIPSIM_POINT_1=0.75
CLIPSIM_POINT_2=0.82
CLIPSIM_POINT_3=0.9
CLIPSIM_POINT_4=0.97


def clip_inter_frame(model, processor, frame_path_list: List[str]) -> float:
    device = model.device
    sim_list = []

    for i in range(len(frame_path_list) - 1):
        img1 = Image.open(frame_path_list[i]).convert("RGB")
        img2 = Image.open(frame_path_list[i + 1]).convert("RGB")

        inputs_1 = processor(images=img1, return_tensors="pt").to(device)
        inputs_2 = processor(images=img2, return_tensors="pt").to(device)

        feat_1 = model.get_image_features(**inputs_1).flatten()
        feat_2 = model.get_image_features(**inputs_2).flatten()

        sim = F.cosine_similarity(feat_1, feat_2, dim=0).item()
        sim_list.append(sim)

    return float(np.mean(sim_list)) if sim_list else None


def clip_sim_output(model_or_process, video_path: str, prompt: str = "") -> float:
    model=model_or_process[0]
    processor=model_or_process[1]

    frame_paths = extract_video_frame_imgs(video_path, fps=2)
    score = clip_inter_frame(model, processor, frame_paths)
    if score < CLIPSIM_POINT_1:
        score = 1
    elif score >= CLIPSIM_POINT_1 and score < CLIPSIM_POINT_2:
        score = 2
    elif score >= CLIPSIM_POINT_2 and score < CLIPSIM_POINT_3:
        score = 3
    elif score >= CLIPSIM_POINT_3 and score < CLIPSIM_POINT_4:
        score = 4
    else:
        score=5
    return score
