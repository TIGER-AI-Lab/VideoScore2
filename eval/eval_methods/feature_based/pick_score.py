import numpy as np
import torch
from PIL import Image
from typing import List
from eval_methods.utils import extract_video_frame_imgs


def pick_score_output(model_or_process, video_path: str, prompt: str = "") -> float:
    frame_paths = extract_video_frame_imgs(video_path)
    if not frame_paths:
        return None
    model=model_or_process[0]
    processor=model_or_process[1]
    device=model_or_process[2]
    
    pil_images = [Image.open(p).convert("RGB") for p in frame_paths]
    
    all_scores = []

    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i + batch_size]

            image_inputs = processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)

            text_inputs = processor(
                text=[prompt] * len(batch_images),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)

            # get normalized embeddings
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            # compute score (PickScore inner product with logit_scale)
            scores = (model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)).cpu().tolist()
            all_scores.extend(scores)

    if not all_scores:
        return None

    return float(np.mean(all_scores))
