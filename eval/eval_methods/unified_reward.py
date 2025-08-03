import torch
import copy
import warnings
from PIL import Image
from typing import Tuple
import os
import re
import random

import sys, os
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

        images, frames_loaded = self._load_video(video_path, num_video_frames, loader_fps, fps, frame_count)
        image_sizes = []
        for i in range(len(images)):
            images[i] = images[i].resize((512, 512))
            image_sizes.append(images[i].size)

        image_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
  
        question = (
            "<image>\n" * len(images) +
            "Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the frames of a given video and see the text prompt for generating the video.\n"
            "Then give scores from 5 different dimensions:\n"
            "(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color\n"
            "(2) temporal consistency, the consistency of objects or humans in video\n"
            "(3) dynamic degree, the degree of dynamic changes\n"
            "(4) text-to-video alignment, the alignment between the text prompt and the video content\n"
            "(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge\n\n"
            "For each dimension, output a number from [1,2,3,4], \nin which '1' means 'Bad', '2' means 'Average', '3' means 'Good', \n'4' means 'Real' or 'Perfect' (the video is like a real video)\n"
            "Finally, based on above 5 dimensions, assign a score from 1.0 to 4.0 after 'Final Score:'\n"
            "Here is an output example:\n"
            "visual quality: 4\ntemporal consistency: 4\ndynamic degree: 3\ntext-to-video alignment: 1\nfactual consistency: 2\nFinal Score: 6\n\n"
            "**Note: In the example above, scores are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given video.**\n"
            f"Your task is provided as follows: Text Prompt: [{user_prompt}]"
        )

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        print(input_ids.shape)
        
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

        pattern = r"Final Score:\s*([\d.]+)"
        match = re.search(pattern, output_text, re.IGNORECASE)

        if match:
            final_score = float(match.group(1))
        else:
            final_score = None

        return final_score, final_score, final_score, output_text
