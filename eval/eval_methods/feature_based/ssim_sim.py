import os
import subprocess
from typing import List
from PIL import Image
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity as ssim
from eval_methods.utils import extract_video_frame_imgs

SSIM_POINT_1=0.6
SSIM_POINT_2=0.7
SSIM_POINT_3=0.8
SSIM_POINT_4=0.9

def ssim_inter_frame(frame_path_list: List[str]) -> float:
    ssim_list = []

    for i in range(len(frame_path_list) - 1):
        frame_1 = np.array(Image.open(frame_path_list[i]).convert("RGB"))
        frame_2 = np.array(Image.open(frame_path_list[i + 1]).convert("RGB"))

        gray_1 = color.rgb2gray(frame_1)
        gray_2 = color.rgb2gray(frame_2)

        ssim_val, _ = ssim(
            gray_1, gray_2,
            full=True,
            data_range=gray_2.max() - gray_2.min()
        )
        ssim_list.append(ssim_val)

    return float(np.mean(ssim_list)) if ssim_list else None


def ssim_sim_output(model, video_path: str, prompt: str = "") -> float:
    frame_paths = extract_video_frame_imgs(video_path, fps=2)
    score = ssim_inter_frame(frame_paths)
    if score < SSIM_POINT_1:
        score = 1
    elif score >= SSIM_POINT_1 and score < SSIM_POINT_2:
        score = 2
    elif score >= SSIM_POINT_2 and score < SSIM_POINT_3:
        score = 3
    elif score >= SSIM_POINT_3 and score < SSIM_POINT_4:
        score = 4
    else:
        score=5
    return score
