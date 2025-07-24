import os
import numpy as np
import torch
from PIL import Image
from typing import List
from pypiqe import piqe
import tempfile
import cv2
from eval_methods.utils import extract_video_frame_imgs

def piqe_output(model_or_process, video_path: str, prompt: str = "") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_paths = extract_video_frame_imgs(video_path)
    
    piqe_list = []

    piqe_list=[]
    for frame_path in frame_paths:
        frame=np.array(Image.open(frame_path))
        piqe_score, _,_,_ = piqe(frame)
        piqe_list.append(piqe_score)
    piqe_avg=np.mean(piqe_list)
       
    return piqe_avg if piqe_list else None
