import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from eval_methods.utils import extract_video_frame_imgs

MAX_LENGTH = 77
DEFAULT_FPS = 4

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

        avg_score = float(np.mean(cos_sim_list)) if cos_sim_list else 0.0

    return avg_score
