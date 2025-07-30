from transformers import AutoModel
from string import Template

MJ_EVAL_TEMP=Template("""As a professional 'Text-to-Video' quality assessor, your task is to determine whether the generated video will be preferred by humans. Please analyze step by step and provide a rating from the scale: {'Extremely Poor', 'Very Poor', 'Poor', 'Below Average','Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Outstanding'}, where 'Extremely Poor' is the worst and 'Outstanding' is the best. This time, please evaluate based on the $category of the video. $category is defined as:
$description.
Do not analyze, and must give a rating. You cannot refuse to answer.
The assessor must directly output the evaluation in the following format: Now, proceed with
evaluating the video based on the prompt description provided. The prompt is: $caption
""")

model_name="MJ-Bench/MJ-VIDEO-2B"


import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL2VideoEvaluator:
    def __init__(self,
                 model_name: str = "OpenGVLab/InternVL2-2B",
                 input_size: int = 448,
                 num_segments: int = 8,
                 max_num: int = 1,
                 dtype=torch.bfloat16):
        self.model_name = model_name
        self.input_size = input_size
        self.num_segments = num_segments
        self.max_num = max_num
        self.dtype = dtype
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        config = AutoConfig.from_pretrained("OpenGVLab/InternVL2-2B", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        return model, tokenizer

    def _build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def _get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def _dynamic_preprocess(self, image, image_size=448, max_num=1, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        transform = self._build_transform(self.input_size)
        frame_indices = self._get_index(bound, fps, max_frame, num_segments=self.num_segments)
        pixel_values_list, num_patches_list = [], []

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img_tiles = self._dynamic_preprocess(img, image_size=self.input_size, max_num=self.max_num)
            tensor_tiles = [transform(tile) for tile in img_tiles]
            tensor_stack = torch.stack(tensor_tiles)
            pixel_values_list.append(tensor_stack)
            num_patches_list.append(tensor_stack.shape[0])

        pixel_values = torch.cat(pixel_values_list).to(self.dtype).cuda()
        return pixel_values, num_patches_list

    def evaluate_video(self,
                       user_prompt: str,
                       video_path: str,
                       kwargs: Optional[dict] = None) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        pixel_values, num_patches_list = self._load_video(video_path)
        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
        full_prompt = video_prefix + user_prompt

        res, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )
        pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
        match = re.search(pattern, res, re.DOTALL | re.IGNORECASE)
        if match:
            v_score_model = int(match.group(1))
            t_score_model = int(match.group(2))
            p_score_model = int(match.group(3))
        else:
            v_score_model = t_score_model = p_score_model = None
        return v_score_model, t_score_model, p_score_model, res

