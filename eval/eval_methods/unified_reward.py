import torch
import copy
import warnings
from PIL import Image
from typing import Tuple
import os
import re
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils_unified_reward")))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import opencv_extract_frames
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX

class eval_UnifiedReward:
    def __init__(self, 
                 model_path: str = "CodeGoat24/UnifiedReward-7b", 
                 model_name: str = "llava_qwen", 
                 device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, None, model_name, device_map=device)
        self.model.eval()
        self.conv_template = "qwen_1_5"
        warnings.filterwarnings("ignore")

    def _load_video(self, video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

    def evaluate_video(self, user_prompt: str, video_path: str, kwargs: dict = None):
        warnings.filterwarnings("ignore")
        kwargs = kwargs or {}
        num_video_frames = kwargs.get("num_video_frames", 8)
        loader_fps = kwargs.get("loader_fps", 0.0)
        infer_fps = kwargs.get("infer_fps", 2.0)
        frame_count = kwargs.get("frame_count", None)
        max_tokens = kwargs.get("max_tokens", 4096)

        images, frames_loaded = self._load_video(video_path, num_video_frames, loader_fps, infer_fps, frame_count)
        image_sizes = []
        for i in range(len(images)):
            images[i] = images[i].resize((512, 512))
            image_sizes.append(images[i].size)

        image_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
          
        question = "<image>\n" * len(images) + user_prompt

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        with torch.cuda.amp.autocast():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_tokens
            )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(output_text)
        
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = float(match.group(1))
            t_score_model = float(match.group(2))
            p_score_model = float(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None
        return v_score_model, t_score_model, p_score_model, output_text