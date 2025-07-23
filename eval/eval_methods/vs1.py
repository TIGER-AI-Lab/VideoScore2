import torch
from PIL import Image
import numpy as np
import av
from typing import List
from transformers import AutoProcessor
from transformers import AutoConfig
from mantis.models.idefics2 import Idefics2ForSequenceClassification

MAX_NUM_FRAMES = 48
ROUND_DIGIT = 3

VS2_VS1_SCALE=1.25


vs1_dimensions = [
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency"
]


class eval_VideoScore1:
    def __init__(self, model_name="TIGER-Lab/VideoScore-v1.1"):
        self.model_name = model_name
        self.model, self.processor = self.load_model_processor(model_name)

    def load_model_processor(self, model_name):
        processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, processor
    
    def _read_video_frames(self, video_path: str, max_frames: int = MAX_NUM_FRAMES) -> List[Image.Image]:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
        else:
            indices = np.arange(total_frames)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
        return frames

    def evaluate_video(self, user_prompt: str, video_path: str, kwargs: dict) -> str:
        frames = self._read_video_frames(video_path, max_frames=MAX_NUM_FRAMES)

        eval_prompt = user_prompt
        eval_prompt += "<image> " * len(frames)

        flatten_images = [Image.open(f) if isinstance(f, str) else f for f in frames]
        inputs = self.processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        num_aspects = logits.shape[-1]
        scores = [round(logits[0, i].item(), ROUND_DIGIT) for i in range(num_aspects)]
        scores=[min(5, max(1, round(s*VS2_VS1_SCALE))) for s in scores]
        print(scores)
        
        res=[f"(1) visual quality: {scores[0]}\n (2) text-to-video alignment: {scores[3]}\n (3) physical/common-sense consistency: {scores[4]}\n"]
        return res
