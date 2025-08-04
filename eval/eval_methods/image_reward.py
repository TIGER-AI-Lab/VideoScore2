import os
import cv2
from typing import List
from ImageReward import load as load_image_reward
from PIL import Image
import tempfile


class eval_ImageReward:
    def __init__(self, model_name_or_path: str = "ImageReward-v1.0"):
        self.model = load_image_reward(model_name_or_path)

    def _extract_frames_from_video(
        self, video_path: str, fps: float = 1.0, max_frames: int = 16
    ) -> List[str]:

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(int(video_fps / fps), 1)

        saved_paths = []
        frame_idx = 0
        save_count = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or save_count >= max_frames:
                    break
                if frame_idx % interval == 0:
                    img_path = os.path.join(tmp_dir, f"frame_{save_count}.jpg")
                    cv2.imwrite(img_path, frame)
                    saved_paths.append(img_path)
                    save_count += 1
                frame_idx += 1

            cap.release()
            image_list = [Image.open(p).convert("RGB") for p in saved_paths]

        return image_list

    def evaluate_video(
        self,
        prompt: str,
        video_path: str,
        kwargs: dict,
    ):
        fps=kwargs.get("fps",2.0)
        max_frames=kwargs.get("max_frames",16)
        image_list = self._extract_frames_from_video(video_path, fps, max_frames)

        rewards = self.model.score(prompt, image_list)
        if len(rewards) == 0:
            return 0.0
        score = sum(rewards) / len(rewards)
        return score, score, score, None
