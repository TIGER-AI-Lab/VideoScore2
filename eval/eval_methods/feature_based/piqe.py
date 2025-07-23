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

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        score, _, _, _ = piqe(tensor)
        piqe_list.append(score.item())    
    return float(np.mean(piqe_list)) if piqe_list else None
