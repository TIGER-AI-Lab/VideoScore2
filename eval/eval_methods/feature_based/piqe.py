import os
import numpy as np
import torch
from PIL import Image
from typing import List
from pypiqe import piqe
import tempfile
import cv2
from eval_methods.utils import extract_video_frame_imgs

PIQE_POINT_1=15
PIQE_POINT_2=25
PIQE_POINT_3=35
PIQE_POINT_4=50

def piqe_output(model_or_process, video_path: str, prompt: str = "") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_paths = extract_video_frame_imgs(video_path)
    
    piqe_list = []

    for frame_path in frame_paths:
        frame=np.array(Image.open(frame_path))
        piqe_score, _,_,_ = piqe(frame)
        piqe_list.append(piqe_score)
    score=np.mean(piqe_list)
    if piqe_list==[]:
        return None
    if score < PIQE_POINT_1:
        score = 1
    elif score >= PIQE_POINT_1 and score < PIQE_POINT_2:
        score = 2
    elif score >= PIQE_POINT_2 and score < PIQE_POINT_3:
        score = 3
    elif score >= PIQE_POINT_3 and score < PIQE_POINT_4:
        score = 4
    else:
        score=5
    return score 
