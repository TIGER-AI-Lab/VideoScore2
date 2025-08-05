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


class eval_Q_Insight:
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
        self.score_format = "The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor and 5 representing excellent. Return the final answer in JSON format with the following keys: \"rating\": The score."
        
        self.gen_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_new_tokens=1024,
        )

    def _evaluate_image(self, image: Image.Image, query_prompt: str) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "frame.jpg")
            image.save(img_path)
            message = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_prompt},
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

        return score, output_text

    def _extract_video_frames(self, video_path: str, fps: float = 2.0, max_frames: int = 16) -> List[Image.Image]:
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
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.score_prompt_dim1 = "The dimension 'visual quality' cares about the image's visual and optical propertities, including resolution, overall clarity, local blurriness, correctness of brightness/contrast, and any other factors the affect the watching experience. What is your overall rating on the visual quality of this picture?"+self.score_format
        self.score_prompt_dim2 = f"The caption for this images is {prompt}. What is your overall rating on the text-to-image alignment of this picture?"+self.score_format
        self.score_prompt_dim3 = "The dimension 'physical/common-sense consistency' mainly examines whether there are any violations of common sense, physical laws, or any other aspects in the image that appear strange or unnatural. What is your overall rating on the physical consistency of this picture?"+self.score_format
        
        fps = kwargs.get("infer_fps", 2.0)
        max_frames = kwargs.get("max_frames", 16)
        frames = self._extract_video_frames(video_path, fps=fps, max_frames=max_frames)

        dim_scores = []
        dim_output_text = []
        for query in [self.score_prompt_dim1,self.score_prompt_dim2,self.score_prompt_dim3]:
            frame_scores = []
            for frame in frames:
                score, output_text = self._evaluate_image(frame,query)
                if score is not None:
                    frame_scores.append(score)
            if not frame_scores:
                dim_scores.append(0.0)
            else:
                dim_scores.append(sum(frame_scores) / len(frame_scores))
            # only save the output text of the last frame for simplicity
            dim_output_text.append(output_text)
        
        
        all_dim_output=f"(1) Visual: {dim_output_text[0]}; (2) Text alignment: {dim_output_text[2]}; (3) Physical/Common-sense Consistency: {dim_output_text[2]}."
        
        return dim_scores[0], dim_scores[1], dim_scores[2], all_dim_output
