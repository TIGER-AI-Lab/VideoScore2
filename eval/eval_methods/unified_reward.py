import torch
import copy
import warnings
from PIL import Image
from typing import Tuple
from eval_methods.unified_reward_utils.llava.model.builder import load_pretrained_model
from eval_methods.unified_reward_utils.llava.mm_utils import opencv_extract_frames
from eval_methods.unified_reward_utils.llava.mm_utils import tokenizer_image_token
from eval_methods.unified_reward_utils.llava.conversation import conv_templates
import random


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

    def _load_video_frames(self, video_path: str, num_video_frames: int = 8, loader_fps: float = 0.0, 
                           fps: float = None, frame_count: int = None) -> Tuple[list, int]:
        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            print(f"[WARNING] Error processing {video_path}: {e}")
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames
            frames_loaded = 0
        return pil_imgs, frames_loaded

    def evaluate_video(self, prompt: str, video_path: str, kwargs: dict = None) -> Tuple[float, float, float, str]:
        kwargs = kwargs or {}
        num_video_frames = kwargs.get("num_video_frames", 8)
        loader_fps = kwargs.get("loader_fps", 0.0)
        fps = kwargs.get("fps", None)
        frame_count = kwargs.get("frame_count", None)

        images, frames_loaded = self._load_video_frames(video_path, num_video_frames, loader_fps, fps, frame_count)

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
            "Finally, based on above 5 dimensions, assign a float score from 1 to 4 after 'Final Score:'\n"
            "Here is an output example:\n"
            "visual quality: 4\ntemporal consistency: 4\ndynamic degree: 3\ntext-to-video alignment: 1\nfactual consistency: 2\nFinal Score: 2.67\n\n"
            "**Note: In the example above, scores are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given video.**\n"
            f"Your task is provided as follows: Text Prompt: [{prompt}]"
        )

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, 0, return_tensors="pt").unsqueeze(0).to(self.device)

        with torch.cuda.amp.autocast():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096
            )
            text_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
        final_score = self._parse_final_score(text_output)
        return final_score, final_score, final_score, text_output

    def _parse_final_score(self, text: str) -> float:
        import re
        match = re.search(r"Final Score:\s*(\d+(\.\d+)?)", text)
        if match:
            return float(match.group(1))
        return 0.0
