import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import cv2


def _extract_frames(video_path: str, infer_fps: float = 2.0, max_frames: int = 16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(max(video_fps // infer_fps, 1))
    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        frame_idx += 1

    cap.release()
    return frames


class eval_DeQA:
    def __init__(self, model_name: str = "zhiyuanyou/DeQA-Score-Mix3"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def evaluate_video(
        self,
        user_prompt: str,  # ignored
        video_path: str,
        kwargs: dict = None,
    ) -> tuple[float, float, float, str]:
        kwargs = kwargs or {}
        infer_fps = kwargs.get("infer_fps", 2.0)
        max_frames = kwargs.get("max_frames", 16)

        if not os.path.exists(video_path):
            raise ValueError(f"Video does not exist: {video_path}")

        try:
            frames = _extract_frames(video_path, infer_fps=infer_fps, max_frames=max_frames)
            if not frames:
                raise ValueError("No frames extracted from video.")
            scores = self.model.score(frames).tolist()
            score = sum(scores) / len(scores)
            return score, score, score, None
        except Exception as e:
            print(f"{e}")
            return None, None, None, None
        
        
