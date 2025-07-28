import os
import json
import torch
from collections.abc import Mapping
from eval_methods.utils_video_reward.prompt_template import build_prompt 
from eval_methods.utils_video_reward.vision_process import process_vision_info 
from eval_methods.utils_video_reward.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from eval_methods.utils_video_reward.utils import load_model_from_checkpoint
from eval_methods.utils_video_reward.data import DataConfig
from eval_methods.utils_video_reward.train_reward import create_model_and_processor


class VideoVLMRewardInference:
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = self.load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )

        model, processor, _ = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        model, _ = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()
        model.to(device)

        self.device = device
        self.model = model
        self.processor = processor
        self.data_config = data_config
        self.inference_config = inference_config

    def load_configs_from_json(self, config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        del config_dict["data_config"]["meta_data"]
        del config_dict["data_config"]["data_dir"]
        return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
            config_dict.get("inference_config", None)

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        return self._prepare_input(inputs)

    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [[{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": max_pixels,
                        "fps": fps,
                        "sample_type": self.data_config.sample_type,
                    },
                    {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                ],
            }] for video_path, prompt in zip(video_paths, prompts)]
        else:
            chat_data = [[{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": max_pixels,
                        "nframes": num_frames,
                        "sample_type": self.data_config.sample_type,
                    },
                    {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                ],
            }] for video_path, prompt in zip(video_paths, prompts)]

        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return self._prepare_inputs(batch)

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(return_dict=True, **batch)["logits"]
        rewards = [{'VQ': r[0].item(), 'MQ': r[1].item(), 'TA': r[2].item()} for r in rewards]
        for r in rewards:
            if use_norm:
                r.update(self._norm(r))
            r['Overall'] = r['VQ'] + r['MQ'] + r['TA']
        return rewards


class eval_VideoReward:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.model = VideoVLMRewardInference(
            load_from_pretrained=model_path,
            device=device,
            dtype=dtype
        )

    def evaluate_video(self, user_prompt: str, video_path: str, kwargs: dict) -> dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        infer_fps = kwargs.get("infer_fps", None)
        max_pixels = kwargs.get("max_pixels", None)
        use_norm = kwargs.get("use_norm", True)

        with torch.no_grad():
            rewards = self.model.reward(
                video_paths=[video_path],
                prompts=[user_prompt],
                fps=infer_fps,
                max_pixels=max_pixels,
                use_norm=use_norm
            )
            return rewards[0]["VQ"], rewards[0]["TA"], None, str(rewards[0])
