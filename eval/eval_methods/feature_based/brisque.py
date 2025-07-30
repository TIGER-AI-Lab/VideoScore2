import os
import numpy as np
import torch
from PIL import Image
from typing import List
import tempfile
import cv2
from brisque import BRISQUE
from eval_methods.utils import extract_video_frame_imgs

BRISQUE_POINT_1=10
BRISQUE_POINT_2=23
BRISQUE_POINT_3=36
BRISQUE_POINT_4=50

def brisque_output(model, video_path: str, prompt: str = "") -> float:
    frame_paths = extract_video_frame_imgs(video_path)

    brisque_list=[]
    for frame_path in frame_paths:
        frame=Image.open(frame_path)
        brisque_score=BRISQUE().score(frame)
        brisque_list.append(brisque_score)
    if brisque_list==[]:
        return None
    score=np.mean(brisque_list)
    if score < BRISQUE_POINT_1:
        score = 1
    elif score >= BRISQUE_POINT_1 and score < BRISQUE_POINT_2:
        score = 2
    elif score >= BRISQUE_POINT_2 and score < BRISQUE_POINT_3:
        score = 3
    elif score >= BRISQUE_POINT_3 and score < BRISQUE_POINT_4:
        score = 4
    else:
        score=5
    return score