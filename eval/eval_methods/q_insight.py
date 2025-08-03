import os
import torch
import cv2
import json
import tempfile
import re
from PIL import Image
from typing import List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info


class eval_QInsight:
    def __init__(
        self,
        model_path: str = "ByteDance/Q-Insight",
        subfolder: str = "score_degradation",
        device: str = "cuda:0",
        seed: int = 42,
    ):
        set_seed(seed)
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
            subfolder=subfolder
        )
        self.processor = AutoProcessor.from_pretrained(model_path, subfolder=subfolder)
        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
            "i.e., <think> reasoning process here </think><answer> answer here </answer>"
        )
        self.score_prompt = (
            "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
            "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. "
            "Return the final answer in JSON format with the following keys: \"rating\": The score."
        )
        self.gen_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_new_tokens=1024,
        )

    def _evaluate_image(self, image: Image.Image) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "frame.jpg")
            image.save(img_path)
            message = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.score_prompt},
                        {"type": "image", "image": f"{img_path}"},
                    ],
                },
            ]

            text = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
            image_inputs, video_inputs = process_vision_info([message])
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    generation_config=self.gen_config,
                    use_cache=True,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            match = re.search(r'"rating"\s*:\s*([0-9.]+)', output_text)
            if match:
                score = float(match.group(1))
            else:
                score = None

        return score, score, score, output_text

    def _extract_video_frames(self, video_path: str, fps: float = 2.0, max_frames: int = 16) -> List[Image.Image]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(int(video_fps / fps), 1)

        frame_idx = 0
        save_count = 0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or save_count >= max_frames:
                break
            if frame_idx % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                save_count += 1
            frame_idx += 1

        cap.release()
        return frames

    def evaluate_video(self, prompt: str, video_path: str, kwargs: dict):
        fps = kwargs.get("infer_fps", 2.0)
        max_frames = kwargs.get("max_frames", 16)
        frames = self._extract_video_frames(video_path, fps=fps, max_frames=max_frames)

        scores = []
        for frame in frames:
            score, _, _, _ = self._evaluate_image(frame)
            if score is not None:
                scores.append(score)

        if not scores:
            return 0.0, 0.0, 0.0, None
        avg_score = sum(scores) / len(scores)
        return avg_score, avg_score, avg_score, f"Q-Insight video avg score: {avg_score:.4f}"
