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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_paths = extract_video_frame_imgs(video_path)

    brisque = BRISQUE().to(device)
    brisque_list = []

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        score = brisque(img_tensor)
        brisque_list.append(score.item())

    return float(np.mean(brisque_list)) if brisque_list else None