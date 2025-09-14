import os
import re
import torch
import pandas as pd
from collections import defaultdict

from peft import LoraConfig, get_peft_model
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils_video_phy2")))
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


PROMPT_SA = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video match the description: \"{caption}\"? "
    "Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.\n"
    "AI: "
)

PROMPT_PHYSICS = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video adhere to the physical laws? "
    "Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations.\n"
    "AI: "
)

PROMPT_RULE = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does the video follow the physical rule: \"{rule}\"?\n"
    "Choose 0 if not, 1 if valid, or 2 if indeterminate. \n"
    "AI: "
)

class eval_VideoPhy2:
    def __init__(self, 
                 checkpoint: str,
                 lora_checkpoint: str = None,
                 num_frames: int = 32):
        self.num_frames = num_frames
        self.model, self.processor, self.tokenizer = self.load_model(checkpoint, lora_checkpoint)

    def load_model(self, checkpoint, lora_checkpoint=None):
        tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
        image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
        processor = MplugOwlProcessor(image_processor, tokenizer)

        model = MplugOwlForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={'': 'cpu'}
        )
        model.eval()

        if lora_checkpoint:
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
                inference_mode=True, 
                r=32, 
                lora_alpha=32, 
                lora_dropout=0.05
            )
            model = get_peft_model(model, peft_config)
            with open(lora_checkpoint, 'rb') as f:
                ckpt = torch.load(f, map_location="cpu")
            try:
                model.load_state_dict(ckpt)
            except:
                ckpt = self._modify_keys(ckpt)
                model.load_state_dict(ckpt)
        
        model = model.to("cuda").to(torch.bfloat16)
        return model, processor, tokenizer

    def _modify_keys(self, state_dict):
        new_state_dict = defaultdict()
        pattern = re.compile(r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight')
        for key, value in state_dict.items():
            if pattern.match(key):
                key = key.split('.')
                key.insert(-1, 'base_layer')
                key = '.'.join(key)
            new_state_dict[key] = value
        return new_state_dict

    def _run_single_prompt(self, video_path: str, prompt: str) -> tuple[int, str]:
        num_map = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
        }

        inputs = self.processor(
            text=[prompt],
            videos=[video_path],
            num_frames=self.num_frames,
            return_tensors='pt'
        )
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        res = self.model.generate(
            **inputs,
            do_sample=False,
            top_k=1,
            temperature=0.001,
            max_length=1024,
        )
        output = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True).lower().strip()

        score = None
        for key, val in num_map.items():
            if key in output:
                score = val
                break
        if score is None:
            digits = ''.join([c for c in output if c.isdigit()])
            score = int(digits) if digits and int(digits) in num_map.values() else 0
            print(f"[Warning] Could not parse output '{output}'. Defaulting to {score}.")
        return score, output

    def evaluate_video(self, caption: str, video_path: str, kwargs: dict) -> tuple[int, int, str, str]:
        """
        Returns: (sa_score, pc_score, sa_output_text, pc_output_text)
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video does not exist: {video_path}")

        sa_prompt = PROMPT_SA.format(caption=caption)
        pc_prompt = PROMPT_PHYSICS

        sa_score, sa_output = self._run_single_prompt(video_path, sa_prompt)
        pc_score, pc_output = self._run_single_prompt(video_path, pc_prompt)

        return None, sa_score, pc_score, f"sa_output: \n{sa_output}\npc_output: \n{pc_output}"