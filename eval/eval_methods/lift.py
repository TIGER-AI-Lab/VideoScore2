import os
import re
import torch
import warnings
import cv2
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils_lift")))
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import (
    opencv_extract_frames,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from llava.model.builder import load_pretrained_model
from llava.conversation import SeparatorStyle, conv_templates
from llava.utils import disable_torch_init


class eval_LiFT:
    def __init__(self, model_path: str):
        disable_torch_init()
        if "13b" in model_path:
            model_base = "Efficient-Large-Model/VILA1.5-13b"
            conv_mode = "vicuna_v1"
        elif "40b" in model_path:
            model_base = "Efficient-Large-Model/VILA1.5-40b"
            conv_mode = "hermes-2"
        else:
            raise ValueError("model_name not supported")
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, self.model_name, model_base
        )
        self.conv_mode = conv_mode or self._infer_conv_mode(self.model_name)

        self.question_types = [
            "semantic consistency",
            "fidelity issues",
            "motion issues",
        ]
        self.question_prompts = [
            " Please identify the semantic consistency issues in this video, focusing on the alignment between the text and the visual content. Specifically, point out any discrepancies regarding the subject (e.g., person or animal), quantity, color, scene description, style, and other relevant aspects. For this video, the text prompt is: [[prompt]]",
            " Please identify the fidelity issues in this video, assessing the realism of the content, including people, animals, and other objects. Highlight any problems such as missing limbs, deformed hands or faces, or other unrealistic elements.",
            " Please identify the motion issues in this video, focusing on the continuity and smoothness of actions, as well as their coherence and adherence to the laws of physics.",
        ]
        warnings.filterwarnings("ignore")

    def _infer_conv_mode(self, model_name: str):
        if "llama-2" in model_name.lower():
            return "llava_llama_2"
        elif "v1" in model_name.lower():
            return "llava_v1"
        elif "mpt" in model_name.lower():
            return "mpt"
        else:
            return "llava_v0"

    def evaluate_video(
        self,
        t2v_prompt: str,
        video_path: str,
        kwargs: dict = None
    ) -> tuple[str, str, str, str]:
        kwargs = kwargs or {}
        num_video_frames = kwargs.get("num_video_frames", 8)
        infer_fps = kwargs.get("infer_fps", 2.0)
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p", None)
        num_beams = kwargs.get("num_beams", 1)
        max_new_tokens = kwargs.get("max_new_tokens", 512)

        try:
            images, _ = opencv_extract_frames(video_path, num_video_frames, infer_fps)
        except Exception as e:
            print(e)
            return None, None, None, None

        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        scores = []
        raw_outputs = []
        
        
        for q_type,question_prompt in zip(self.question_types,self.question_prompts):
            if q_type in ["semantic consistency"]:
                qs = question_prompt.replace("[[prompt]]",t2v_prompt)
            else:
                qs = question_prompt
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, qs)
            elif DEFAULT_IMAGE_TOKEN not in qs:
                qs = ((image_token_se if self.model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN) + "\n") * len(images) + qs

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[images_tensor],
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            print(output_ids.shape)
            
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if output_text.endswith(stop_str):
                output_text = output_text[:-len(stop_str)].strip()
            if "Good. " in output_text:
                scores.append(3)
            elif "Normal. " in output_text:
                scores.append(2)
            elif "Bad. " in output_text:
                scores.append(1)
            else:
                scores.append(None)
            raw_outputs.append(output_text)

        return scores[0], scores[1], scores[2], "\n\n".join(raw_outputs)
