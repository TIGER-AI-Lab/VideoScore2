import os
import numpy as np
import torch
from PIL import Image
from typing import List
import tempfile
import cv2
from brisque import BRISQUE
from eval_methods.utils import extract_video_frame_imgs

def brisque_output(model, video_path: str, prompt: str = "") -> float:
    frame_paths = extract_video_frame_imgs(video_path)

    brisque_list=[]
    for frame_path in frame_paths:
        frame=Image.open(frame_path)
        brisque_score=BRISQUE().score(frame)
        brisque_list.append(brisque_score)
    brisque_avg=np.mean(brisque_list)
    
    return brisque_avg if brisque_list else None