import os
import cv2
import ast
import torch
import tempfile
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class eval_AIGVE_MACS:
    def __init__(self, model_name="xiaoliux/AIGVE-MACS", device="cuda:0", dtype=torch.bfloat16):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            # attn_implementation="flash_attention_2"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    def _extract_video_frames(self, video_path, fps=1, max_frames=16):
        """
        Extract frames from video at specified fps, return list of frame image paths
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps // fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_paths = []
        tmp_dir = tempfile.mkdtemp(prefix="aigve_frames_")
        frame_count = 0
        saved = 0

        while cap.isOpened() and saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                save_path = os.path.join(tmp_dir, f"frame_{saved:03d}.png")
                cv2.imwrite(save_path, frame)
                frame_paths.append(save_path)
                saved += 1
            frame_count += 1

        cap.release()
        return frame_paths

    def evaluate_video(self, user_prompt: str, video_path: str, kwargs: dict) -> str | None:
        fps = kwargs.get("infer_fps", 2)
        max_tokens = kwargs.get("max_tokens", 1500)

        frame_paths = self._extract_video_frames(video_path, fps=fps)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "fps": fps,
                        "video": frame_paths,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        out_dict=ast.literal_eval(output_text)
        tq=out_dict["technical_quality"]["score"]
        phy=out_dict["physics"]["score"]
        return tq, None, phy, output_text
