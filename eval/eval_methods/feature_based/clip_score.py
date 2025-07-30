import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from eval_methods.utils import extract_video_frame_imgs

MAX_LENGTH = 77
DEFAULT_FPS = 4

CLIP_POINT_1=0.25
CLIP_POINT_2=0.283
CLIP_POINT_3=0.316
CLIP_POINT_4=0.35


def clip_score_output(model_or_process, video_path: str, prompt: str) -> float:
    model=model_or_process[0]
    processor=model_or_process[1]
    device=model_or_process[2]
    
    frame_paths = extract_video_frame_imgs(video_path, fps=DEFAULT_FPS)
    input_text = processor(text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)

    with torch.no_grad():
        text_feat = model.get_text_features(**input_text).flatten()

        cos_sim_list = []
        for frame_path in tqdm(frame_paths, desc="CLIP frame sim"):
            image = Image.open(frame_path).convert("RGB")
            image_input = processor(images=image, return_tensors="pt").to(device)
            image_feat = model.get_image_features(**image_input).flatten()
            cos_sim = F.cosine_similarity(text_feat, image_feat, dim=0).item()
            cos_sim_list.append(cos_sim)

        score = float(np.mean(cos_sim_list)) if cos_sim_list else 0.0
    if score < CLIP_POINT_1:
        score = 1
    elif score >= CLIP_POINT_1 and score < CLIP_POINT_2:
        score = 2
    elif score >= CLIP_POINT_2 and score < CLIP_POINT_3:
        score = 3
    elif score >= CLIP_POINT_3 and score < CLIP_POINT_4:
        score = 4
    else:
        score=5
    return score
